#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "signal.h"
#include <stdint.h>

#define inline __inline

// #define NDEBUG

#ifdef _OPENMP
  #include "omp.h"
#else 
   //inline int omp_get_thread_num() {return 0;}
   //inline int omp_get_num_threads() {return 1;}
   //inline int omp_get_max_threads() {return 1;}
#endif


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// integrator parameters

#define DT 0.1
#define GAMMA 0.5
#define MASS 200

// intra strand parameters
#define R_INTRA 0.513
#define K_INTRA 200

// inter-strand parameters
#define R_INTER 1.1
#define K_INTER 100
#define E_INTER 1
#define E_INTER_a 0.0	// unpaired 'bp'
#define E_INTER_A 0.95
#define E_INTER_C 1.05
#define CUT_INTER 1.2

// angle parameters
float K_ANGLE = 5.0;
#define E_ANGLE 0.1

#ifdef LADDER
	#define THETA0 1.570796 //radians is 90 degrees
#else
	#define THETA0 1.387 //radians is 80.2 degrees
#endif

// dihedral parameters
float K_DIHED = 1.0;
#define E_DIHED 0.1

// non-bonded parameters
#define N_HC 4
#define M_HC 2
#define R_HC_INTRA 0.5
#define R_HC_INTER 1.0
#define CUT_NEIGH 1.5
#define MAX_NEIGH 2568

#define PITCH 10.6

// global energy variables
float intraE, interE, angleE, dihedralE, intraHardE, interHardE;

// position, velocity, force
float (*x)[3];
float (*v)[3];
float (*f)[3];

#ifdef SEQ
// A or C base defined for sequence
char *BaseType;
#endif

// displacement tracking for neighbour search
float (*xRef)[3];

// neighbour list
int (*neigh)[MAX_NEIGH+1];

// association data
int *isBound;

int N,N2,stopRunning=0, Junc_index;	//record the base index at the junction during folding
uint64_t t;
int N_bpfix = 0;
int N_bpfree = 0;
int N_bpfree2 = 0;

#ifdef DELETEBP
int N_bkup = 0, N_temp = 0;
#endif
#ifdef ETE
int Win_Size = 0;
#endif
float f_PULL = 0.0f;

char * restartfile[2] = {"restart1","restart2"};
int restartid=0;

// ladder case does not have any twist
#ifdef LADDER
static const float PHI_1 = 180.0;
static const float PHI_2 = 180.0;
#else
static const float PHI_1 = 104.94;
static const float PHI_2 = 104.94; 
#endif

static float sin1,cos1,sin2,cos2;

unsigned long initseed[4];
unsigned long length = 4;

#include "helper.c"
#include "generators.c"
#include "vtf.c"
#include "neighbour.c"
#include "langevin.c"
#include "restart.c"

void graceful_exit(int signo)
{
	if (signo == SIGINT) printf("\nReceived SIGINT ");
	//if (signo == SIGUSR1) printf("\nReceived SIGUSR1 ");
	//if (signo == SIGUSR2) printf("\nReceived SIGUSR2 ");
	 
	printf ("at timestep %llu, will abort after this step.",t);
	fflush(stdout);
	stopRunning = 1;
}

void main(int argc, char ** argv ) {
	uint64_t nsteps = atoi(argv[2]);
	float temperature = atof (argv[3]);
	
	int target = atof (argv[13]);
		f_PULL = atof (argv[14]);
	
	//print("N_bpfix and N_bpfree are necessary parameters. You can input 0 and 0 for them if no bp will be fixed or free.\n");
	// number of base-pairs being fixed at the head
	N_bpfix = atof (argv[10]);
	// number of base-pairs being free at the tail and the loop
	N_bpfree = atof (argv[11]);
	N_bpfree2 = atof (argv[12]);
	
	int i, rebuildCount = 0, rebuildDelay = 101; 
	//uint32_t  jsr = (unsigned)time(0);
	float n,m;
	float sigma;
	#ifdef SEQ
		char tempbase[2];
	#endif
	FILE *traj   	= initVTF("traj.vtf");
	FILE *energy	= fopen("energy.dat", "a"); 
	FILE *bubbles	= fopen("bubbles.dat", "a");


	signal(SIGINT,  graceful_exit);
	//signal(SIGUSR1, graceful_exit);
	//signal(SIGUSR2, graceful_exit);

	printf("MD simulation of a coarse grained DNA model.\nCompiled in ");
	#ifdef CIRCULAR 
	printf("CIRCULAR ");
	#endif
	#ifdef LADDER 
	printf("LADDER ");
	#else
	printf("COIL ");
	#endif
	printf("mode ");
	#ifdef _OPENMP
	printf("with OpenMP, running %d threads.\n", omp_get_max_threads());
	#else
	printf("without OpenMP.\n");
	#endif
	
	//jsr *= atof (argv[4]);
	//printf("random seed %x\n", jsr);
	
	if (argc<10)  { printf("I need 9 or 10 parameters: N, nsteps, temperature, rand1,rand2,rand3,rand4, K_ANGLE, K_DIHED, (Target for Autostop version).\n"); exit(1); }

	// get the parameters :  N, NSTEPS, TEMPERATURE
	N = atoi(argv[1]);
	N2 = 2*N;
	Junc_index = N - N_bpfix;	// 4 bp fixed, so the junction is at N - 4 for the initial state
	#ifdef DELETEBP
	N_bkup = N;
	#endif
	#ifdef ETE
	Win_Size = atoi(argv[15]);
	#endif
	

	// seeding a random stream for each thread
	//r4_nor_setup ( kn, fn, wn );
	// init gen_rand with 4 seeds
	initseed[0] = atof (argv[4]);
	initseed[1] = atof (argv[5]);
	initseed[2] = atof (argv[6]);
	initseed[3] = atof (argv[7]);
	initseed[0] %= 9000;
	initseed[1] %= 9000;
	initseed[2] %= 9000;
	initseed[3] %= 9000;

	init_by_array(initseed, length);
	genrand_real1();
	
	// input K_ANGLE and K_DIHED
	K_ANGLE = atof (argv[8]);
	K_DIHED = atof (argv[9]);
  	
//#ifdef _OPENMP
//	// allocate seed array 
//	int max_threads = omp_get_max_threads();
//	seed = malloc ( max_threads * sizeof ( uint32_t ) );
//	// generate seeds for each thread
//	for (i=0; i < max_threads; i++ ) 
//		seed[i] = shr3_seeded ( &jsr );
//#else
//	seed = jsr;
//#endif


	// the order of hardcore repulsion and associated sigma
	n=N_HC;
	m=M_HC;
	sigma = pow((m/n), (1.0/(n-m)));
	printf ("Hardcore based on mie %d-%d, sigma is %f\n",N_HC,M_HC, sigma);
	sigma2_intra = R_HC_INTRA * R_HC_INTRA * sigma*sigma;
	sigma2_inter = R_HC_INTER * R_HC_INTER * sigma*sigma;
	printf ("intra sigma is %.3f and inter sigma is %.3f.\n",sqrt(sigma2_intra), sqrt(sigma2_inter));

	// calculate sin and cos shifts for dihedrals once 
	sin1 = sin(PHI_1/180.*M_PI);
	cos1 = cos(PHI_1/180.*M_PI);
	sin2 = sin(PHI_2/180.*M_PI);
	cos2 = cos(PHI_2/180.*M_PI);

	//allocate global arrays 
	
	x = malloc(sizeof((*x))*2*N);
	v = malloc(sizeof((*v))*2*N);
	f = malloc(sizeof((*f))*2*N);
	
	#ifdef SEQ
		// readin sequence
		BaseType = malloc(sizeof((*BaseType))*N);
		FILE *ReadSeq = fopen("sequence.dat", "r");
		
		for(i = 0 ; i < N ; i++){
			fgets(tempbase,2,ReadSeq);
			BaseType[i] = tempbase[0];
			//printf("%d %c\n", i, BaseType[i]);
		}
		fclose(ReadSeq);
	#endif

	xRef	= malloc(sizeof((*xRef))*2*N);
	neigh	= malloc(sizeof((*neigh))*2*N);
	isBound	= malloc(sizeof(int)*N);

	// force neighbour rebuild at the first step
	
	zero(xRef);
	xRef[0][0] = CUT_NEIGH;
	
#ifndef CIRCULAR	
	// first and last beads are initially (and always) bound 
	//isBound[0]=1;
	isBound[N-4]=1;
	isBound[N-3]=1;
	isBound[N-2]=1;
	isBound[N-1]=1;
#endif

	
	// if there is no restart file, 
	// do minimization via langevin at 0 temperature
	if ( !readRestart("restart") ) {
		FILE *minim = initVTF("minim.vtf"); 

		#if defined LADDER
		genLadder();
		#elif defined CIRCULAR
		genCircCoil(PITCH);
		#else
		genCoil(PITCH);
		#endif
		
		zero(f);
		zero(v);
		t=0;

		printf ("Minimizing...");

		do {
			if (t%1000 ==0) writeVTF(minim);	
			calcNeigh();
			integrateLangevin(0.005,0);
			t++;
		} while (maxForce(f) > 0.5 && t < 100000 );

		printf ("done after %llu steps.\n",t);
		fclose(minim);
		
		//write minimized config to restart file
		writeRestart(restartfile[restartid]);
	}

	// open files in append mode 

	//FILE *traj   	= initVTF("traj.vtf");
	//FILE *energy	= fopen("energy.dat", "a"); 
	//FILE *bubbles	= fopen("bubbles.dat", "a");
	#ifdef NDEBUG			
	FILE *neighCount= fopen("neigh.dat", "a");
	#endif

	//force neighbour rebuild at first step
	xRef[0][0] = CUT_NEIGH;

	printf("N=%d, T=%.3f, beginning run for %llu steps..\n", N, temperature, nsteps);
	fflush(stdout);

	/******* MAIN LOOP BEGINS HERE ********/
	#ifdef AUTOSTOP
	for (t=0; t<nsteps * 100000000; t++){
	#else
	for (t=0; t<nsteps * 10000; t++){
	#endif
	
	
		// neighbour list rebuild with delay
		if ( calcNeigh() ) { 
			rebuildCount ++ ;
			#ifdef NDEBUG			
			printNeighCount(neighCount); fflush(neighCount);
			printNeigh();
			#endif
		}

		// integration
		integrateLangevin(DT, temperature);
			

		// print bubble matrix
		#ifdef DELETEBP
		N_temp = N;
		N = N_bkup;
		#endif
		if (t % 10000 == 0)   printBubble(bubbles);
		#ifdef DELETEBP
		N = N_temp;
		#endif

		
		//write trajectory and energy
		#ifdef DELETEBP
		N_temp = N;
		N = N_bkup;
		#endif
		if (t % 10000 == 0) {
			writeVTF(traj);
			printEnergy(energy);
		}
		#ifdef DELETEBP
		N = N_temp;
		#endif
		
		//write restart file and flush buffers
		if (t % 1000000 == 0) {
		  restartid = 1 - restartid;
			writeRestart(restartfile[restartid]);
			fflush(bubbles); fflush(traj); fflush(energy);
			printf("\rWritten %s at step %llu.",restartfile[restartid],t); fflush(stdout);
		}
		#ifdef AUTOSTOP
		if (t % 10000 == 0){
			stopRunning = 1;
			for (i=0; i<N-N_bpfix; i++){
				if(isBound[i] != target)
					stopRunning = 0;
			}
			//if (stopRunning == 1) break;
			//stopRunning = 0;
		}
		#endif



		// stop running if necessary;
		if (stopRunning) break;
	}
	
	/******* MAIN LOOP ENDS HERE ******/

	//write final restart
	#ifdef DELETEBP
	N = N_bkup;
	#endif
	printBubble(bubbles);
	writeVTF(traj);
	printEnergy(energy);
	writeRestart("restart");
	
	printf("\nStopped at step %llu.\nNeighbour list was rebuilt %d times.\n",t,rebuildCount);

	//close all files
	free(x); free(v); free(f); free(xRef);
	#ifdef SEQ
		free(BaseType);
	#endif
	free(isBound); free(neigh);
	#ifdef _OPENMP
	free(seed);
	#endif
	fclose(traj); fclose(energy); fclose(bubbles);
	#ifdef NDEBUG			
	fclose(neighCount);
	#endif

}
