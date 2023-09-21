#ifndef GLUT_WRAP_H
#define GLUT_WRAP_H

#ifdef HAVE_FREEGLUT
#  include <GL/freeglut.h>
#elif defined __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glut.h>
#endif

#ifndef GLAPIENTRY
#define GLAPIENTRY
#endif

#endif /* ! GLUT_WRAP_H */
