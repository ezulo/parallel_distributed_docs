#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__
#pragma once
inline int min( int a, int b ) { return a < b ? a : b; }
inline int max( int a, int b ) { return a > b ? a : b; }

#include <vector>
using namespace std;

//
//  saving parameters
//
const int NSTEPS = 1000;
const int SAVEFREQ = 10;

//
// particle data structure
//
typedef struct
{
  double x;
  double y;
  double vx;
  double vy;
  double ax;
  double ay;
} particle_t;

typedef vector<particle_t*> Specks;
typedef vector<vector<Specks> > CellMatrix;
typedef struct {
        int x,y;
} Point;
//
//  timing routines
//
double read_timer( );

//
//  simulation routines
//
void set_size( int n );
void init_particles( int n, particle_t *p );
void apply_force( particle_t &particle, particle_t &neighbor , double *dmin, double *davg, int *navg);
void move( particle_t &p );


//
//  I/O routines
//
FILE *open_save( char *filename, int n );
void save( FILE *f, int n, particle_t *p );

//
//  argument processing routines
//
int find_option( int argc, char **argv, const char *option );
int read_int( int argc, char **argv, const char *option, int default_value );
char *read_string( int argc, char **argv, const char *option, char *default_value );

int cellcount();
void cellmatrix(CellMatrix&);
void updater(particle_t *w,CellMatrix &p, int len);
Point cellindex(particle_t&);
void updateparticles(int a, int b, particle_t*, CellMatrix&);
void apply_force_more(particle_t* ,CellMatrix&, double *, double *, int *);
void printPoint(Point);
void printParticle(particle_t*);
bool isthesame(particle_t*, particle_t*);
void clearParticles(CellMatrix&);
void clearParticles(int h, int g, CellMatrix&);
void pfromr(int, int, Specks*, CellMatrix&);

template <typename T>
T clamp(T in, T min, T max)
{
        return std::min(std::max(in,min), max);
}


#endif