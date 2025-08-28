{
  description = "Toolbox and trackers for object pose-estimation";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        { pkgs, ... }:
        {
          devShells.default =
            with pkgs;
            let
              libs = [
                glib
                libGL
                libjpeg
                zlib
              ];
            in
            mkShell {
              packages = [
                uv
              ]
              ++ libs;
              env = {
                LD_LIBRARY_PATH = lib.makeLibraryPath libs;
                CFLAGS = lib.concatMapStringsSep " " (
                  x: "-I${lib.getInclude x}/include -L${lib.getLib x}/lib"
                ) libs;
              };
            };
        };
    };
}
