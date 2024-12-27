#include "math.h"
//#include<iostream>
#include<stdio.h>
#include<stdlib.h>

////////////////////////random generator///////////////////////

 /* Period parameters */  
#define NNNN 624
#define MMMM 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UMASK 0x80000000UL /* most significant w-r bits */
#define LMASK 0x7fffffffUL /* least significant r bits */
#define MIXBITS(u,v) ( ((u) & UMASK) | ((v) & LMASK) )
#define TWIST(u,v) ((MIXBITS(u,v) >> 1) ^ ((v)&1UL ? MATRIX_A : 0UL))

static unsigned long randstate[NNNN]; /* the array for the state vector  */
static int randleft = 1;
static int randinitf = 0;
static unsigned long *randnext;

/* initializes randstate[NNNN] with a seed */
void init_genrand(unsigned long s)
{
    int j;
    randstate[0]= s & 0xffffffffUL;
    for (j=1; j<NNNN; j++) {
        randstate[j] = (1812433253UL * (randstate[j-1] ^ (randstate[j-1] >> 30)) + j); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array randstate[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        randstate[j] &= 0xffffffffUL;  /* for >32 bit machines */
    }
    randleft = 1; randinitf = 1;
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
void init_by_array(unsigned long init_key[], unsigned long key_length)
{
    int i, j, k;
    init_genrand(19650218UL);
    i=1; j=0;
    k = (NNNN>key_length ? NNNN : key_length);
    for (; k; k--) {
        randstate[i] = (randstate[i] ^ ((randstate[i-1] ^ (randstate[i-1] >> 30)) * 1664525UL))
          + init_key[j] + j; /* non linear */
        randstate[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++; j++;
        if (i>=NNNN) { randstate[0] = randstate[NNNN-1]; i=1; }
        if (((unsigned long)j)>=key_length) j=0;
    }
    for (k=NNNN-1; k; k--) {
        randstate[i] = (randstate[i] ^ ((randstate[i-1] ^ (randstate[i-1] >> 30)) * 1566083941UL))
          - i; /* non linear */
        randstate[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i>=NNNN) { randstate[0] = randstate[NNNN-1]; i=1; }
    }

    randstate[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */ 
    randleft = 1; randinitf = 1;
}

static void randnext_state(void)
{
    unsigned long *p=randstate;
    int j;

    /* if init_genrand() has not been called, */
    /* a default initial seed is used         */
    if (randinitf==0) init_genrand(5489UL);

    randleft = NNNN;
    randnext = randstate;
    
    for (j=NNNN-MMMM+1; --j; p++) 
        *p = p[MMMM] ^ TWIST(p[0], p[1]);

    for (j=MMMM; --j; p++) 
        *p = p[MMMM-NNNN] ^ TWIST(p[0], p[1]);

    *p = p[MMMM-NNNN] ^ TWIST(p[0], randstate[0]);
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(void)
{
    unsigned long y;

    if (--randleft == 0) randnext_state();
    y = *randnext++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

/* generates a random number on [0,0x7fffffff]-interval */
long genrand_int31(void)
{
    unsigned long y;

    if (--randleft == 0) randnext_state();
    y = *randnext++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return (long)(y>>1);
}

/* generates a random number on [0,1]-real-interval */
double genrand_real1(void)
{
    unsigned long y;

    if (--randleft == 0) randnext_state();
    y = *randnext++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return (double)y * (1.0/4294967295.0); 
    /* divided by 2^32-1 */ 
}

/* generates a random number on [0,1)-real-interval */
double genrand_real2(void)
{
    unsigned long y;

    if (--randleft == 0) randnext_state();
    y = *randnext++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return (double)y * (1.0/4294967296.0); 
    /* divided by 2^32 */
}

/* generates a random number on (0,1)-real-interval */
double genrand_real3(void)
{
    unsigned long y;

    if (--randleft == 0) randnext_state();
    y = *randnext++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return ((double)y + 0.5) * (1.0/4294967296.0); 
    /* divided by 2^32 */
}

/* generates a random number on [0,1) with 53-bit resolution*/
double genrand_res53(void) 
{
    unsigned long a=genrand_int32()>>5, b=genrand_int32()>>6; 
    return(a*67108864.0+b)*(1.0/9007199254740992.0); 
}
/* These real versions are due to Isaku Wada, 2002/01/09 added */
////////////////////////random generator///////////////////////
////////////////////////random generator///////////////////////

///////////////////
