#!/bin/bash


################################################################################

# Location of the OSMesa installation:
osmesaprefix="${OSMESA_PREFIX:-/opt/osmesa}"

# Location of the LLVM installation:
llvmprefix="${LLVM_PREFIX:-/opt/llvm}"

################################################################################


# This is a modified version of: https://github.com/devernay/osmesa-install
# The changes are documented in: https://github.com/thodan/bop_renderer/blob/master/README.md


# environment variables used by this script:
# - OSMESA_PREFIX: where to install osmesa (must be writable)
# - LLVM_PREFIX: where llvm is / should be installed
# - LLVM_BUILD: whether to build LLVM (0/1, 0 by default)
# - SILENT_LOG: redirect output and error to log file (0/1, 0 by default)
# - BUILD_OSDEMO: try to compile and run osdemo (0/1, 1 by default)

set -e # Exit immediately if a command exits with a non-zero status
set -u # Treat unset variables as an error when substituting.
#set -x # Print commands and their arguments as they are executed.

# mesa version
mesaversion="${OSMESA_VERSION:-17.1.6}"
# mesa-demos version
demoversion=8.3.0
# glu version
gluversion=9.0.0
# set debug to 1 to compile a version with debugging symbols
debug=0
# set clean to 1 to clean the source directories first (recommended)
clean=1
# number of parallel make jobs, set to 4 by default
mkjobs="${MKJOBS:-4}"
# set osmesadriver to:
# - 1 to use "classic" osmesa resterizer instead of the Gallium driver
# - 2 to use the "softpipe" Gallium driver
# - 3 to use the "llvmpipe" Gallium driver (also includes the softpipe driver, which can
#     be selected at run-time by setting en var GALLIUM_DRIVER to "softpipe")
# - 4 to use the "swr" Gallium driver (also includes the softpipe driver, which can
#     be selected at run-time by setting en var GALLIUM_DRIVER to "softpipe")
osmesadriver=${OSMESA_DRIVER:-4}
# do we want a mangled mesa + GLU ?
mangled=0
# do we want to build the proper LLVM static libraries too? or are they already installed ?
buildllvm="${LLVM_BUILD:-1}"
llvmversion="${LLVM_VERSION:-4.0.1}"
# redirect output and error to log file; exit script on error.
silentlogging="${SILENT_LOG:-0}"
buildosdemo="${BUILD_OSDEMO:-0}"
osname=$(uname)
# This script
scriptdir=$(cd "$(dirname "$0")"; pwd)
# scriptname is the script name without the suffix (if any)
scriptname=$(basename "$0")
scriptname="${scriptname%.*}"
if [ "$silentlogging" = 1 ]; then
    # Exit script on error, redirect output and error to log file. Open log for realtime updates.
    set -e
    exec </dev/null &>"$scriptdir/$scriptname.log"
fi

if [ "$debug" = 1 ]; then
    CFLAGS="${CFLAGS:--g}"
else
    CFLAGS="${CFLAGS:--O3}"
fi
CXXFLAGS="${CXXFLAGS:-${CFLAGS}}"

if [ -z "${CC:-}" ]; then
    CC=gcc
fi
if [ -z "${CXX:-}" ]; then
    CXX=g++
fi

if [ "$osname" = Darwin ]; then
    osver=$(uname -r | awk -F . '{print $1}')
    # Possible $osver values:
    # 9: Mac OS X 10.5 Leopard
    # 10: Mac OS X 10.6 Snow Leopard
    # 11: Mac OS X 10.7 Lion
    # 12: OS X 10.8 Mountain Lion
    # 13: OS X 10.9 Mavericks
    # 14: OS X 10.10 Yosemite
    # 15: OS X 10.11 El Capitan
    # 16: macOS 10.12 Sierra
    # 17: macOS 10.13 High Sierra
    
    if [ "$osver" = 10 ]; then
        # On Snow Leopard (10.6), build universal
        archs="-arch i386 -arch x86_64"
        CFLAGS="$CFLAGS $archs"
        CXXFLAGS="$CXXFLAGS $archs"
    fi
    XCODE_VER=$(xcodebuild -version | sed -e 's/Xcode //' | head -n 1)
    case "$XCODE_VER" in
        4.2*|5.*|6.*|7.*|8.*)
            # clang became the default compiler on Xcode 4.2
            CC=clang
            CXX=clang++
            ;;
    esac

    # Note: the macOS deployment target (used eg for option -mmacosx-version-min=10.<X>) is set
    # in the script from the MACOSX_DEPLOYMENT_TARGET environment variable.
    # To set it from the command-line, use e.g. "env MACOSX_DEPLOYMENT_TARGET=10.8 ../osmesa-install.sh"

    # Similarly, the SDK root can be set using the SDKROOT environment variable, as in
    # "env MACOSX_DEPLOYMENT_TARGET=10.8 SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.12.sdk ../osmesa-install.sh"

    if [ "$osmesadriver" = 4 ]; then
        #     "swr" (aka OpenSWR) is not supported on macOS,
        #     https://github.com/OpenSWR/openswr/issues/2
        #     https://github.com/OpenSWR/openswr-mesa/issues/11
        osmesadriver=3
    fi
    if [ "$osver" = 10 ]; then
        # On Snow Leopard, if using the system's gcci with libstdc++, build with llvm 3.4.2.
        # If using libc++ (see https://trac.macports.org/wiki/LibcxxOnOlderSystems), compile
        # everything with clang-5.0
        if [ -f /opt/local/etc/macports/macports.conf ] && grep -q -e '^cxx_stdlib.*libc\+\+' /opt/local/etc/macports/macports.conf; then
            if [[ $(type -P clang-mp-5.0) ]]; then
		CC=clang-mp-5.0
		CXX=clang++-mp-5.0
		OSDEMO_LD="clang++-mp-5.0 -stdlib=libc++"
	    else
		echo "Error: Please install clang 5 using the following command:"
		echo "sudo port install clang-5.0"
	    fi
        else
            # This project is affected by a bug in Apple's gcc driver driver that was fixed in the apple-gcc42 port:
            # https://github.com/macports/macports-ports/blob/master/lang/apple-gcc42/files/driverdriver-num_infiles.patch
            # Use that or clang.
            if [[ $(type -P clang-mp-3.4) ]]; then
                CC=clang-mp-3.4
                CXX=clang++-mp-3.4
            elif [[ $(type -P gcc-apple-4.2) ]]; then
                CC=gcc-apple-4.2
                CXX=g++-apple-4.2
            fi
            if [ -z "${LLVM_VERSION+x}" ]; then
                llvmversion=3.4.2
            fi
        fi
    fi
fi

# tell curl to continue downloads and follow redirects
curlopts="-L -C -"
srcdir="$scriptdir"

echo "Mesa buid options:"
if [ "$debug" = 1 ]; then
    echo "- debug"
else
    echo "- release, non-debug"
fi
if [ "$mangled" = 1 ]; then
    echo "- mangled (all function names start with mgl instead of gl)"
else
    echo "- non-mangled"
fi

if [ "$osmesadriver" = 1 ]; then
    echo "- classic osmesa software renderer"
elif [ "$osmesadriver" = 2 ]; then
    echo "- softpipe Gallium renderer"
elif [ "$osmesadriver" = 3 ]; then
    echo "- llvmpipe Gallium renderer"
    if [ "$buildllvm" = 1 ]; then
        echo "- also build and install LLVM $llvmversion in $llvmprefix"
    fi
elif [ "$osmesadriver" = 4 ]; then
    echo "- swr Gallium renderer"
    if [ "$buildllvm" = 1 ]; then
        echo "- also build and install LLVM $llvmversion in $llvmprefix"
    fi
else
    echo "Error: osmesadriver must be 1, 2, 3 or 4"
    exit
fi
if [ "$clean" = 1 ]; then
    echo "- clean sources"
fi

if [ -n "${MACOSX_DEPLOYMENT_TARGET+x}" ]; then
    echo "- compile for deployment on macOS $MACOSX_DEPLOYMENT_TARGET (mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET)"
fi
if [ -n "${SDKROOT+x}" ]; then
    echo "- OSX SDK root is $SDKROOT"
fi


# On MacPorts, building Mesa requires the following packages:
# sudo port install xorg-glproto xorg-libXext xorg-libXdamage xorg-libXfixes xorg-libxcb

llvmlibs=
if [ ! -d "$osmesaprefix" ] || [ ! -w "$osmesaprefix" ]; then
    echo "Error: $osmesaprefix does not exist or is not user-writable, please create $osmesaprefix and make it user-writable"
    exit
fi
if [ "$osmesadriver" = 3 ] || [ "$osmesadriver" = 4 ]; then
    # see also https://wiki.qt.io/Cross_compiling_Mesa_for_Windows
    if [ "$buildllvm" = 1 ]; then
        if [ ! -d "$llvmprefix" ] || [ ! -w "$llvmprefix" ]; then
            echo "Error: $llvmprefix does not exist or is not user-writable, please create $llvmprefix and make it user-writable"
            exit
        fi
        # LLVM must be compiled with RRTI, see https://bugs.freedesktop.org/show_bug.cgi?id=90032
        if [ "$clean" = 1 ]; then
            rm -rf llvm-${llvmversion}.src
        fi

        archsuffix=xz
        xzcat=xzcat
        if [ $llvmversion = 3.4.2 ]; then
            archsuffix=gz
            xzcat="gzip -dc"
        fi
        # From Yosemite (14) gunzip can decompress xz files - but only if containing a tar archive.
        if [ "$osname" = Darwin ] && [ "$osver" -ge 14 ]; then
            xzcat="gzip -dc"
        fi
        if [ ! -f llvm-${llvmversion}.src.tar.$archsuffix ]; then
            echo "* downloading LLVM ${llvmversion}..."
            # the llvm we server doesnt' allow continuing partial downloads
            curl $curlopts -O "http://www.llvm.org/releases/${llvmversion}/llvm-${llvmversion}.src.tar.$archsuffix"
        fi
        $xzcat llvm-${llvmversion}.src.tar.$archsuffix | tar xf -
        cd llvm-${llvmversion}.src
        echo "* building LLVM..."
        cmake_archflags=
        if [ $llvmversion = 3.4.2 ] && [ "$osname" = Darwin ] && [ "$osver" = 10 ]; then
            if [ "$debug" = 1 ]; then
                debugopts="--disable-optimized --enable-debug-symbols --enable-debug-runtime --enable-assertions"
            else
                debugopts="--enable-optimized --disable-debug-symbols --disable-debug-runtime --disable-assertions"
            fi
            # On Snow Leopard, build universal
            # and use configure (as macports does)
            # workaround a bug in Apple's shipped gcc driver-driver
            if [ "$CXX" = "g++" ]; then
                echo "static int ___ignoreme;" > tools/llvm-shlib/ignore.c
            fi
            env CC="$CC" CXX="$CXX" REQUIRES_RTTI=1 UNIVERSAL=1 UNIVERSAL_ARCH="i386 x86_64" ./configure --prefix="$llvmprefix" \
                --enable-bindings=none --disable-libffi --enable-shared --disable-static --enable-jit --enable-pic \
                --enable-targets=host --disable-profiling \
                --disable-backtraces \
                --disable-terminfo \
                --disable-zlib \
                $debugopts
            env REQUIRES_RTTI=1 UNIVERSAL=1 UNIVERSAL_ARCH="i386 x86_64" make -j"${mkjobs}" install
            echo "* installing LLVM..."
            env REQUIRES_RTTI=1 UNIVERSAL=1 UNIVERSAL_ARCH="i386 x86_64" make install
        else
            cmakegen="Unix Makefiles" # can be "MSYS Makefiles" on MSYS
            cmake_archflags=""
            llvm_patches=""
            if [ "$osname" = Darwin ] && [ "$osver" = 10 ]; then
                # On Snow Leopard, build universal
                cmake_archflags="-DCMAKE_OSX_ARCHITECTURES=i386;x86_64"
                # Proxy for eliminating the dependency on native TLS
                # http://trac.macports.org/ticket/46887
                #cmake_archflags="$cmake_archflags -DLLVM_ENABLE_BACKTRACES=OFF" # flag was added to the common flags below, we don't need backtraces anyway

                # https://llvm.org/bugs/show_bug.cgi?id=25680
                #configure.cxxflags-append -U__STRICT_ANSI__
            fi

            if [ "$osname" = Darwin ]; then
                # Redundant - provided for older compilers that do not pass this option to the linker
                # Address xcode/cmake error: compiler appears to require libatomic, but cannot find it.
                cmake_archflags="-DLLVM_ENABLE_LIBCXX=ON"
                if [ "$osver" -ge 12 ]; then
                    # From Mountain Lion onward. We are only building 64bit arch.
                    cmake_archflags="$cmake_archflags -DCMAKE_OSX_ARCHITECTURES=x86_64"
                fi
                # https://cmake.org/cmake/help/v3.0/variable/CMAKE_OSX_DEPLOYMENT_TARGET.html
                if [ -n "${MACOSX_DEPLOYMENT_TARGET+x}" ]; then
                    cmake_archflags="$cmake_archflags -DCMAKE_OSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"
                fi
                # https://cmake.org/cmake/help/v3.0/variable/CMAKE_OSX_SYSROOT.html
                if [ -n "${SDKROOT+x}" ]; then
                    cmake_archflags="$cmake_archflags -DCMAKE_OSX_SYSROOT=$SDKROOT"
                fi
            fi

            case "$osname" in
                 Msys*|MSYS*|MINGW*)
                     cmakegen="MSYS Makefiles"
                     #cmake_archflags="-DLLVM_ENABLE_CXX1Y=ON" # is that really what we want???????
                     cmake_archflags="-DLLVM_USE_CRT_DEBUG=MTd -DLLVM_USE_CRT_RELEASE=MT"
                     llvm_patches="msys2_add_pi.patch"
                     ;;
            esac

            # Apply ORC patch on LLVM 4.x and 5.x
            if [ ${llvmversion:0:2} = 4. ] || [ ${llvmversion:0:2} = 5. ]; then
                llvm_patches="0001-Fix-return-type-in-ORC-readMem-client-interface.patch :$llvm_patches"
            fi

            for i in $llvm_patches; do
                if [ -f "$srcdir"/patches/llvm-$llvmversion/$i ]; then
                    echo "* applying patch $i"
                    patch -p1 -d . < "$srcdir"/patches/llvm-$llvmversion/$i
                fi
            done
            mkdir build
            cd build
            if [ "$debug" = 1 ]; then
                debugopts="-DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INCLUDE_TESTS=ON -DLLVM_INCLUDE_EXAMPLES=ON"
            else
                debugopts="-DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF"
            fi

            env CC="$CC" CXX="$CXX" REQUIRES_RTTI=1 cmake -G "$cmakegen" .. -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_INSTALL_PREFIX="${llvmprefix}" \
                -DLLVM_TARGETS_TO_BUILD="host" \
                -DLLVM_ENABLE_RTTI=ON \
                -DLLVM_REQUIRES_RTTI=ON \
                -DBUILD_SHARED_LIBS=ON \
                -DBUILD_STATIC_LIBS=OFF \
                -DLLVM_ENABLE_FFI=OFF \
                -DLLVM_BINDINGS_LIST=none \
                -DLLVM_ENABLE_PEDANTIC=OFF \
                -DLLVM_INCLUDE_TESTS=OFF \
                -DLLVM_ENABLE_BACKTRACES=OFF \
                -DLLVM_ENABLE_TERMINFO=OFF \
                -DLLVM_ENABLE_ZLIB=OFF \
                $debugopts $cmake_archflags
            env REQUIRES_RTTI=1 make -j"${mkjobs}"
            echo "* installing LLVM..."
            env REQUIRES_RTTI=1 make install
            cd ..
        fi
        cd ..
    fi
    llvmconfigbinary=
    case "$osname" in
        Msys*|MSYS*|MINGW*)
            llvmconfigbinary="$llvmprefix/bin/llvm-config.exe"
            ;;
        *)
            llvmconfigbinary="$llvmprefix/bin/llvm-config"
            ;;
    esac
    # Check if llvm installed
    if [ ! -x "$llvmconfigbinary" ]; then
        # could not find installation.
        if [ "$buildllvm" = 0 ]; then
            # advise user to turn on automatic download, build and install option
            echo "Error: $llvmconfigbinary does not exist, set script variable buildllvm=\${LLVM_BUILD:-0} from 0 to 1 to automatically download and install llvm."
        else
            echo "Error: $llvmconfigbinary does not exist, please install LLVM with RTTI support in $llvmprefix"
            echo " download the LLVM sources from llvm.org, and configure it with:"
            echo " env CC=$CC CXX=$CXX cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$llvmprefix -DBUILD_SHARED_LIBS=ON -DLLVM_ENABLE_RTTI=1 -DLLVM_REQUIRES_RTTI=1 -DLLVM_ENABLE_PEDANTIC=0 $cmake_archflags"
            echo " env REQUIRES_RTTI=1 make -j${mkjobs}"
        fi
        exit
    fi
    llvmcomponents="engine mcjit"
    if [ "$debug" = 1 ]; then
        llvmcomponents="$llvmcomponents mcdisassembler"
    fi
    llvmlibs=$("${llvmconfigbinary}" --ldflags --libs $llvmcomponents)
    if "${llvmconfigbinary}" --help 2>&1 | grep -q system-libs; then
        llvmlibsadd=$("${llvmconfigbinary}" --system-libs)
    else
        # on old llvm, system libs are in the ldflags
        llvmlibsadd=$("${llvmconfigbinary}" --ldflags)
    fi
    llvmlibs="$llvmlibs $llvmlibsadd"
fi

if [ "$clean" = 1 ]; then
    rm -rf "mesa-$mesaversion" "mesa-demos-$demoversion" "glu-$gluversion"
fi

if [ ! -f "mesa-${mesaversion}.tar.gz" ]; then
    echo "* downloading Mesa ${mesaversion}..."
    echo "ftp://ftp.freedesktop.org/pub/mesa/mesa-${mesaversion}.tar.gz"
    curl $curlopts -O "ftp://ftp.freedesktop.org/pub/mesa/older-versions/17.x/mesa-${mesaversion}.tar.gz" || curl $curlopts -O "ftp://ftp.freedesktop.org/pub/mesa/older-versions/17.x/${mesaversion}/mesa-${mesaversion}.tar.gz"
fi
tar zxf "mesa-${mesaversion}.tar.gz"

# apply patches from MacPorts

echo "* applying patches..."

#add_pi.patch still valid with Mesa 17.0.3
#gallium-once-flag.patch only for Mesa < 12.0.1
#gallium-osmesa-threadsafe.patch still valid with Mesa 17.0.3
#glapi-getproc-mangled.patch only for Mesa < 11.2.2
#install-GL-headers.patch still valid with Mesa 17.0.3
#lp_scene-safe.patch still valid with Mesa 17.0.3
#mesa-glversion-override.patch
#osmesa-gallium-driver.patch still valid with Mesa 17.0.3
#redefinition-of-typedef-nirshader.patch only for Mesa 12.0.x
#scons25.patch only for Mesa < 12.0.1
#scons-llvm-3-9-libs.patch still valid with Mesa 17.0.3
#swr-sched.patch still valid with Mesa 17.0.3
#disable_shader_cache.patch still valid with Mesa 17.1.6 and should be applied on Mavericks and earlier (may be fixed later, check https://trac.macports.org/ticket/54638#comment:8)
#osmesa-gl-DispatchTSD.patch still valid with Mesa 17.1.10
#osmesa-configure-ac.patch still valid with Mesa 17.1.10

PATCHES="\
add_pi.patch \
gallium-once-flag.patch \
gallium-osmesa-threadsafe.patch \
glapi-getproc-mangled.patch \
install-GL-headers.patch \
lp_scene-safe.patch \
mesa-glversion-override.patch \
osmesa-gallium-driver.patch \
redefinition-of-typedef-nirshader.patch \
scons25.patch \
scons-llvm-3-9-libs.patch \
swr-sched.patch \
scons-swr-cc-arch.patch \
msys2_scons_fix.patch \
osmesa-gl-DispatchTSD.patch \
osmesa-configure-ac.patch \
"

if [ "$osname" = Darwin ] && [ "$osver" -lt 14 ]; then
    # See https://trac.macports.org/ticket/54638
    # See https://trac.macports.org/ticket/54643
    PATCHES="$PATCHES disable_shader_cache.patch"
fi

#if mangled, add mgl_export (for mingw)
if [ "$mangled" = 1 ]; then
    PATCHES="$PATCHES mgl_export.patch"
fi

# mingw-specific patches (for maintainability, prefer putting everything in the main patch list)
#if [ "$osname" = "Msys" ] || [ "$osname" = "MINGW64_NT-6.1" ] || [ "$osname" = "MINGW32_NT-6.1" ]; then
#    PATCHES="$PATCHES "
#fi

if [ "$osname" = Darwin ]; then
    # patches for Mesa 12.0.1 from
    # https://github.com/macports/macports-ports/tree/master/x11/mesa/files
    PATCHES="$PATCHES \
    0001-mesa-Deal-with-size-differences-between-GLuint-and-G.patch \
    0002-applegl-Provide-requirements-of-_SET_DrawBuffers.patch \
    0003-glext.h-Add-missing-include-of-stddef.h-for-ptrdiff_.patch \
    5002-darwin-Suppress-type-conversion-warnings-for-GLhandl.patch \
    static-strndup.patch \
    no-missing-prototypes-error.patch \
    o-cloexec.patch \
    patch-include-GL-mesa_glinterop_h.diff \
    "
fi

for i in $PATCHES; do
    if [ -f "$srcdir/patches/mesa-$mesaversion/$i" ]; then
        echo "* applying patch $i"
        patch -p1 -d "mesa-${mesaversion}" < "$srcdir/patches/mesa-$mesaversion/$i"
    fi
done

cd "mesa-${mesaversion}"

echo "* fixing gl_mangle.h..."
# edit include/GL/gl_mangle.h, add ../GLES*/gl[0-9]*.h to the "files" variable and change GLAPI in the grep line to GL_API
gles=
for h in GLES/gl.h GLES2/gl2.h GLES3/gl3.h GLES3/gl31.h GLES3/gl32.h; do
    if [ -f include/$h ]; then
        gles="$gles ../$h"
    fi
done
(cd include/GL; sed -e 's@gl.h glext.h@gl.h glext.h '"$gles"'@' -e 's@\^GLAPI@^GL_\\?API@' -i.orig gl_mangle.h)
(cd include/GL; sh ./gl_mangle.h > gl_mangle.h.new && mv gl_mangle.h.new gl_mangle.h)

echo "* fixing src/mapi/glapi/glapi_getproc.c..."
# functions in the dispatch table sre not stored with the mgl prefix
sed -i.bak -e 's/MANGLE/MANGLE_disabled/' src/mapi/glapi/glapi_getproc.c

echo "* building Mesa..."

case "$osname" in
    Msys*|MSYS*|MINGW*)

        ####################################################################
        # Windows build uses scons

        case "$osname" in
            MINGW64_NT-*)
                scons_machine="x86_64"
                ;;
            *)
                scons_machine="x86"
                ;;
        esac
        scons_cflags="$CFLAGS"
        scons_cxxflags="$CXXFLAGS -std=c++11"
        scons_ldflags="-static -s"
        if [ "$mangled" = 1 ]; then
            scons_cflags="-DUSE_MGL_NAMESPACE"
        fi
        if [ "$debug" = 1 ]; then
            scons_build="debug"
        else
            scons_build="release"
        fi
        if [ "$osmesadriver" = 3 ] || [ "$osmesadriver" = 4 ]; then
            scons_llvm=yes
        else
            scons_llvm=no
        fi
        if [ "$osmesadriver" = 4 ]; then
            scons_swr=1
        else
            scons_swr=0
        fi
        mkdir -p "$osmesaprefix/include" "$osmesaprefix/lib/pkgconfig"
        env LLVM_CONFIG="$llvmconfigbinary" LLVM="$llvmprefix" CFLAGS="$scons_cflags" CXXFLAGS="$scons_cxxflags" LDFLAGS="$scons_ldflags" scons build="$scons_build" platform=windows toolchain=mingw machine="$scons_machine" texture_float=yes llvm="$scons_llvm" swr="$scons_swr" verbose=yes osmesa
        cp "build/windows-$scons_machine/gallium/targets/osmesa/osmesa.dll" "$osmesaprefix/lib/osmesa.dll"
        cp "build/windows-$scons_machine/gallium/targets/osmesa/libosmesa.a" "$osmesaprefix/lib/libMangledOSMesa32.a"
        cp -a "include/GL" "$osmesaprefix/include/" || exit 1
        cat <<EOF > "$osmesaprefix/lib/pkgconfig/osmesa.pc"
prefix=${osmesaprefix}
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: osmesa
Description: Mesa Off-screen Rendering library
Requires:
Version: $mesaversion
Libs: -L\${libdir} -lMangledOSMesa32
Cflags: -I\${includedir}
EOF
        cp $osmesaprefix/lib/pkgconfig/osmesa.pc $osmesaprefix/lib/pkgconfig/gl.pc

        # end of SCons build
        ####################################################################
        ;;
    *)

        ####################################################################
        # Unix builds use configure

        test -f Mafefile && make -j"${mkjobs}" distclean # if in an existing build

        autoreconf -fi

        confopts="\
            --disable-dependency-tracking \
            --disable-static \
            --enable-shared \
            --enable-texture-float \
            --disable-gles1 \
            --disable-gles2 \
            --disable-dri \
            --disable-dri3 \
            --disable-glx \
            --disable-glx-tls \
            --disable-egl \
            --disable-gbm \
            --disable-xvmc \
            --disable-vdpau \
            --disable-omx \
            --disable-va \
            --disable-opencl \
            --disable-shared-glapi \
            --disable-driglx-direct \
            --with-dri-drivers= \
            --with-osmesa-bits=32 \
            --with-egl-platforms= \
            --prefix=$osmesaprefix \
            "

        if [ "$osmesadriver" = 1 ]; then
            # pure osmesa (swrast) OpenGL 2.1, GLSL 1.20
            confopts="${confopts} \
                 --enable-osmesa \
                 --disable-gallium-osmesa \
                 --disable-gallium-llvm \
                 --with-gallium-drivers= \
            "
        elif [ "$osmesadriver" = 2 ]; then
            # gallium osmesa (softpipe) OpenGL 3.0, GLSL 1.30
            confopts="${confopts} \
                 --disable-osmesa \
                 --enable-gallium-osmesa \
                 --disable-gallium-llvm \
                 --with-gallium-drivers=swrast \
            "
        elif [ "$osmesadriver" = 3 ]; then
            # gallium osmesa (llvmpipe) OpenGL 3.0, GLSL 1.30
            confopts="${confopts} \
                 --disable-osmesa \
                 --enable-gallium-osmesa \
                 --enable-gallium-llvm=yes \
                 --with-llvm-prefix=$llvmprefix \
                 --disable-llvm-shared-libs \
                 --with-gallium-drivers=swrast \
            "
        else
            # gallium osmesa (swr) OpenGL 3.0, GLSL 1.30
            confopts="${confopts} \
                 --disable-osmesa \
                 --enable-gallium-osmesa \
                 --with-llvm-prefix=$llvmprefix \
                 --disable-llvm-shared-libs \
                 --with-gallium-drivers=swrast,swr \
            "
        fi

        if [ "$debug" = 1 ]; then
            confopts="${confopts} \
                 --enable-debug"
        fi

        if [ "$mangled" = 1 ]; then
            confopts="${confopts} \
                 --enable-mangling"
            #sed -i.bak -e 's/"gl"/"mgl"/' src/mapi/glapi/gen/remap_helper.py
            #rm src/mesa/main/remap_helper.h
        fi

        if [ "$osname" = Darwin ]; then
            osxflags=""
            if [ "$osver" -ge 12 ]; then
                # From Mountain Lion onward so we are only building 64bit arch.
                osxflags="$osxflags -arch x86_64"
            fi
            if [ -n "${MACOSX_DEPLOYMENT_TARGET+x}" ]; then
                osxflags="$osxflags -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET"
            fi
            if [ -n "${SDKROOT+x}" ]; then
                osxflags="$osxflags -isysroot $SDKROOT"
            fi

            if [ -n "$osxflags" ]; then
                CFLAGS="$CFLAGS $osxflags"
                CXXFLAGS="$CXXFLAGS $osxflags"
            fi
        fi

        env CC="$CC" CXX="$CXX" PTHREADSTUBS_CFLAGS=" " PTHREADSTUBS_LIBS=" " ./configure ${confopts} CC="$CC" CFLAGS="$CFLAGS" CXX="$CXX" CXXFLAGS="$CXXFLAGS"

        make -j"${mkjobs}"

        echo "* installing Mesa..."
        make install

        if [ "$osname" = Darwin ]; then
            # fix the following error:
            #Undefined symbols for architecture x86_64:
            #  "_lp_dummy_tile", referenced from:
            #      _lp_rast_create in libMangledOSMesa32.a(lp_rast.o)
            #      _lp_setup_set_fragment_sampler_views in libMangledOSMesa32.a(lp_setup.o)
            #ld: symbol(s) not found for architecture x86_64
            #clang: error: linker command failed with exit code 1 (use -v to see invocation)
            for f in "$osmesaprefix/lib/"lib*.a; do
                ranlib -c "$f"
            done
        fi

        # End of configure-based build
        ####################################################################
        ;;
esac

cd ..

if [ ! -f glu-${gluversion}.tar.bz2 ]; then
    echo "* downloading GLU ${gluversion}..."
    curl $curlopts -O "ftp://ftp.freedesktop.org/pub/mesa/glu/glu-${gluversion}.tar.bz2"
fi
tar jxf glu-${gluversion}.tar.bz2
cd glu-${gluversion}
echo "* building GLU..."
confopts="\
    --disable-dependency-tracking \
    --disable-static \
    --enable-shared \
    --enable-osmesa \
    --prefix=$osmesaprefix"
if [ "$mangled" = 1 ]; then
    confopts="${confopts} \
     CPPFLAGS=-DUSE_MGL_NAMESPACE"
fi

env PKG_CONFIG_PATH="${PKG_CONFIG_PATH:+${PKG_CONFIG_PATH}:}${osmesaprefix}/lib/pkgconfig" ./configure ${confopts} CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS"
make -j"${mkjobs}"

echo "* installing GLU..."
make install

if [ "$mangled" = 1 ]; then
    mv "$osmesaprefix/lib/libGLU.a" "$osmesaprefix/lib/libMangledGLU.a"
    mv "$osmesaprefix/lib/libGLU.la" "$osmesaprefix/lib/libMangledGLU.la"
    sed -e s/libGLU/libMangledGLU/g -i.bak "$osmesaprefix/lib/libMangledGLU.la"
    sed -e s/-lGLU/-lMangledGLU/g -i.bak "$osmesaprefix/lib/pkgconfig/glu.pc"
fi

if [ "$buildosdemo" = 1 ]; then
# build the demo
cd ..
if [ ! -f mesa-demos-${demoversion}.tar.bz2 ]; then
    echo "* downloading Mesa Demos ${demoversion}..."
    curl $curlopts -O "ftp://ftp.freedesktop.org/pub/mesa/demos/${demoversion}/mesa-demos-${demoversion}.tar.bz2"
fi
tar jxf mesa-demos-${demoversion}.tar.bz2

cd mesa-demos-${demoversion}/src/osdemos
echo "* building Mesa Demo..."
# We need to include gl_mangle.h and glu_mangle.h, because osdemo32.c doesn't include them

INCLUDES="-include $osmesaprefix/include/GL/gl.h -include $osmesaprefix/include/GL/glu.h"
if [ "$mangled" = 1 ]; then
    INCLUDES="-include $osmesaprefix/include/GL/gl_mangle.h -include $osmesaprefix/include/GL/glu_mangle.h $INCLUDES"
    LIBS32="-lMangledOSMesa32 -lMangledGLU"
else
    LIBS32="-lOSMesa32 -lGLU"
fi
if [ -z "${OSDEMO_LD:-}" ]; then
    OSDEMO_LD="$CXX"
fi
if [ "$osname" = Darwin ] || [ "$osname" = Linux ]; then
    # strange, got 'Undefined symbols for architecture x86_64' without zlib for both llvmpipe and softpipe drivers.
    # missing symbols are _deflate, _deflateEnd, _deflateInit_, _inflate, _inflateEnd and _inflateInit
    LIBS32="$LIBS32 -lz"
fi
echo "$OSDEMO_LD $CFLAGS -I$osmesaprefix/include -I../../src/util $INCLUDES  -o osdemo32 osdemo32.c -L$osmesaprefix/lib $LIBS32 $llvmlibs"
if [ "$osname" = Darwin ] || [ "$osname" = Linux ]; then
    allowfail=false
else
    # build on windows may fail because osmesa is a dll but glu is not, thus:
    #G:\msys64\tmp\mingw32\cclAYXEC.o:osdemo32.c:(.text.startup+0x11f): undefined reference to `_imp__mgluNewQuadric@0'
    #G:\msys64\tmp\mingw32\cclAYXEC.o:osdemo32.c:(.text.startup+0x480): undefined reference to `_imp__mgluCylinder@36'
    #G:\msys64\tmp\mingw32\cclAYXEC.o:osdemo32.c:(.text.startup+0x4f8): undefined reference to `_imp__mgluSphere@20'
    #G:\msys64\tmp\mingw32\cclAYXEC.o:osdemo32.c:(.text.startup+0x513): undefined reference to `_imp__mgluDeleteQuadric@4'
    #collect2.exe: error: ld returned 1 exit status
    allowfail=true
fi
$OSDEMO_LD $CFLAGS -I$osmesaprefix/include -I../../src/util $INCLUDES  -o osdemo32 osdemo32.c -L$osmesaprefix/lib $LIBS32 $llvmlibs || ./osdemo32 image.tga || $allowfail
# result is in image.tga
fi # buildosdemo == 1

exit

# Useful information:
# Configuring osmesa 9.2.2:
# http://www.paraview.org/Wiki/ParaView/ParaView_And_Mesa_3D#OSMesa.2C_Mesa_without_graphics_hardware

# MESA_GL_VERSION_OVERRIDE an OSMesa should not be used before Mesa 11.2,
# + patch for earlier versions:
# https://cmake.org/pipermail/paraview/2015-December/035804.html
# patch: http://public.kitware.com/pipermail/paraview/attachments/20151217/4854b0ad/attachment.bin

# llvmpipe vs swrast benchmarks:
# https://cmake.org/pipermail/paraview/2015-December/035807.html

#env MESA_GL_VERSION_OVERRIDE=3.2 MESA_GLSL_VERSION_OVERRIDE=150 ./osdemo32

# Local Variables:
# indent-tabs-mode: nil
# sh-basic-offset: 4
# sh-indentation: 4
