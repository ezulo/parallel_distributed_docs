#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "common.h"
#include <typeinfo>
#include <iostream>
double size;

//
//  tuned constants
//
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

//
//  timer
//
double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

//
//  keep density constant
//
void set_size( int n )
{
    size = sqrt( density * n );
}

//
//  Initialize the particle positions and velocities
//
void init_particles( int n, particle_t *p )
{
    srand48( time( NULL ) );

    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;

    int *shuffle = (int*)malloc( n * sizeof(int) );
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;

    for( int i = 0; i < n; i++ )
    {
        //
        //  make sure particles are not spatially sorted
        //
        int j = lrand48()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];

        //
        //  distribute particles evenly to ensure proper spacing
        //
        p[i].x = size*(1.+(k%sx))/(1+sx);
        p[i].y = size*(1.+(k/sx))/(1+sy);

        //
        //  assign random velocities within a bound
        //
        p[i].vx = drand48()*2-1;
        p[i].vy = drand48()*2-1;
    }
    free( shuffle );
}

//
//  interact two particles
//
void apply_force( particle_t &particle, particle_t &neighbor , double *dmin, double *davg, int *navg)
{

    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
        if (r2 != 0)
        {
           if (r2/(cutoff*cutoff) < *dmin * (*dmin))
              *dmin = sqrt(r2)/cutoff;
           (*davg) += sqrt(r2)/cutoff;
           (*navg) ++;
        }

    r2 = fmax( r2, min_r*min_r );
    double r = sqrt( r2 );



    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

//
//  integrate the ODE
//
void move( particle_t &p )
{
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x  += p.vx * dt;
    p.y  += p.vy * dt;

    //
    //  bounce from walls
    //
    while( p.x < 0 || p.x > size )
    {
        p.x  = p.x < 0 ? -p.x : 2*size-p.x;
        p.vx = -p.vx;
    }
    while( p.y < 0 || p.y > size )
    {
        p.y  = p.y < 0 ? -p.y : 2*size-p.y;
        p.vy = -p.vy;
    }
}

//
//  I/O routines
//
void save( FILE *f, int n, particle_t *p )
{
    static bool first = true;
    if( first )
    {
        fprintf( f, "%d %g\n", n, size );
        first = false;
    }
    for( int i = 0; i < n; i++ )
        fprintf( f, "%g %g\n", p[i].x, p[i].y );
}

//
//  command line option processing
//
int find_option( int argc, char **argv, const char *option )
{
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

char *read_string( int argc, char **argv, const char *option, char *default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}

int cellcount()
{
  return (int) ceil(size/cutoff);
}

void cellmatrix(CellMatrix &particles)
{
        int x = particles.size();
        for(int i = 0; i < x; i++)
        {
          particles[i].resize(x);
        }
}

void updater(particle_t *w, CellMatrix &p, int len)
{
        int z = p.size();
        for(int i = 0; i < z; i++)
        {
         for(int j = 0; j < z; j++)
         {
           p[i][j].clear();
           p[i][j].resize(0);
         }
        }
        for(int i = 0; i < len; i++)
        {
         Point q = cellindex(w[i]);
         p[q.y][q.x].push_back(&w[i]);
        }
}

void updateparticles(int a, int b, particle_t *items, CellMatrix &c)
{
        for(int i = a; i < b; i++)
        {
          Point e = cellindex(items[i]);
          c[e.y][e.x].push_back(&items[i]);
        }
}

void clearParticles(CellMatrix &c)
{
  int h = c.size();
  for(int i = 0; i < h; i++)
  {
    int g = c[i].size();
    for(int j = 0; j < g; j++)
    {
        c[i][j].clear();
        c[i][j].resize(0);
    }
  }
}

void clearParticles(int a, int b, CellMatrix& c)
{
   for(int i = a; i < b; i++)
   {
        int h = c[i].size();
        for(int j = 0; j < h; j++)
        {
          c[i][j].clear();
          c[i][j].resize(0);
        }
   }
}

Point cellindex(particle_t &c)
{
   Point t = {floor(c.x/cutoff), floor(c.y/cutoff)};
   return t;
}

void apply_force_more(particle_t *b, CellMatrix &a, double* dmin, double *davg, int *navg)
{
   double tmp = (*dmin);
   double tmp1 = (*davg);
   int tmp2 = (*navg);
   Point ctr = cellindex(*b);
   int r = cellcount();
   int starta = ctr.y - 1;
   starta = clamp<int>(starta, 0, ctr.y);
   int enda = ctr.y + 1;
   enda = clamp<int>(enda, 0, r - 1);
   int startb = ctr.x - 1;
   startb = clamp<int>(startb, 0, ctr.x);
   int endb = ctr.x + 1;
   endb = clamp<int>(endb, 0, r - 1);
   for(int g = starta; g <= enda; g++)
   {
    for(int h = startb; h <= endb; h++)
    {
        if(!a[g][h].empty())
        {
         Specks neighbors = a[g][h];
         int z = neighbors.size();
         for(int i = 0; i < z; i++)
         {
          particle_t *neighbor = a[g][h][i];
          if(b == neighbor)
          {
           continue;
          }
          apply_force(*b, *neighbor, &tmp, &tmp1, &tmp2);
         }
        }
    }
   }
}

void printPoint(Point q)
{
  printf("Point: (%d, %d)\n", q.y, q.x);
}

void printParticle(particle_t *item)
{
  cout << "particle: " << "(" << item->x << ", " << item->y << ")" << endl;
}

bool isthesame(particle_t *t1, particle_t *t2)
{
  return(t1->x == t2->x && t1->y == t2->y);
}

void pfromr(int start, int end, Specks *items, CellMatrix& things)
{
   for(int j = start; j < end; j++)
   {
     int tmpc = things[j].size();
     for(int k = 0; k < tmpc; k++)
     {
      Specks cells = things[j][k];
      int tmp = cells.size();
      for(int i = 0; i < tmp; i++)
      {
        items->push_back(cells[i]);
      }
     }
   }
}
