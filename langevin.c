#include "dihedral.c"
#include "harmonic.c"
#include "hardcore.c"
#include "harcos.c"
#include "angle.c"
#include "random.h"

float rng_gaussian_real()
{
	float x,y,r;
	
    do{
		x = 2.0 * genrand_real1() - 1.0;
		y = 2.0 * genrand_real1() - 1.0;
		r = x * x + y * y;
    }while(r > 1.0 || r == 0.0);
    
    r = sqrt(-2.0 * log(r) / r);
    //rng->gauss_saved=y*r; /* save second random number */
    //rng->has_saved=1;
    return x*r; /* return first random number */
}



//// used for ziggurat
//static float fn[128];
//static uint32_t kn[128];
//static float wn[128];
//// used for SHR3
//#ifdef _OPENMP
//static uint32_t *seed;
//#else
//static uint32_t seed;
//#endif


#ifdef _OPENMP
// normal random number generator using ziggurat method
static float ziggurat(int thread_num) {  
	
	uint32_t  tmp = seed[thread_num];
      	float random = r4_nor (& tmp , kn, fn, wn );
	seed[thread_num] = tmp;
	return random;
      }
#endif

float rk = M_PI / ( CUT_INTER - R_INTER );

// Velocity Verlet integration with Langevin EoM

static void integrateLangevin(float dt, float temperature) 
{
	const float halfdtgamma = 0.5*GAMMA*dt;
	const float halfdtgammanorm = 1/(1+halfdtgamma);
	const float halfdtMass = 0.5*dt/MASS;
	const float randFmult = sqrt(2*temperature*GAMMA*MASS/dt);

	float c,s; 
	float kMult, dkMult;
	
	// reset energy
	
	intraE=0;
	interE=0;
	angleE=0;
	dihedralE=0;
	intraHardE=0;
	interHardE=0;

	#pragma omp parallel 
	{
	#ifdef _OPENMP	
	int thread_num = omp_get_thread_num();
	int num_threads = omp_get_num_threads();
	#endif

	int i,j,k, ii, ete_J;	//ETE Judge
	float del[3],norm,rsq;
	float inter, x1, x2;
	
	#ifdef BARWIND
	float r, dr, kdr, fmult, x0;
	#endif
	#ifdef HTWOD
	float r, dr, kdr, fmult, x0;
	#endif
	
	#ifdef DELETEBP
	int i1 = 0, i2 = 0;
	float coord1[3],coord2[3];
	
	for (i=N-1; i>=0; i--) {
		j=2*i;
		k=j+1;
		
		del[0] = x[j][0] - x[k][0];
		del[1] = x[j][1] - x[k][1];
		del[2] = x[j][2] - x[k][2];
		
		rsq = del[0]*del[0] + del[1]*del[1] + del[2]*del[2];
		
		if (rsq >= CUT_INTER*CUT_INTER){
			i1 = i + 1;
			break;
		}
	}
	
	if (i1 < N-4){
		//coord1[0] = (x[2 * i1 + 0][0] + x[2 * i1 + 1][0]) / 2.0f;
		//coord1[1] = (x[2 * i1 + 0][1] + x[2 * i1 + 1][1]) / 2.0f;
		//coord1[2] = (x[2 * i1 + 0][2] + x[2 * i1 + 1][2]) / 2.0f;
		//coord2[0] = (x[2 * N - 8][0] + x[2 * N - 7][0]) / 2.0f;
		//coord2[1] = (x[2 * N - 8][1] + x[2 * N - 7][1]) / 2.0f;
		//coord2[2] = (x[2 * N - 8][2] + x[2 * N - 7][2]) / 2.0f;
		//
		//for (i=0; i<=3; i++) {
		//	j=2*(i1+i);
		//	k=j+1;
		//	
		//	x[j][0] = x[2 * N - 8 + 2 * i][0] + coord1[0] - coord2[0];
		//	x[j][1] = x[2 * N - 8 + 2 * i][1] + coord1[1] - coord2[1];
		//	x[j][2] = x[2 * N - 8 + 2 * i][2] + coord1[2] - coord2[2];
		//	x[k][0] = x[2 * N - 7 + 2 * i][0] + coord1[0] - coord2[0];
		//	x[k][1] = x[2 * N - 7 + 2 * i][1] + coord1[1] - coord2[1];
		//	x[k][2] = x[2 * N - 7 + 2 * i][2] + coord1[2] - coord2[2];
		//	
		//}
		
		N = i1 + 4;
		N2 = 2*N;
	}
	
		// this must be defined together with DELETEBP
		#ifdef ADDBP
		if (i1 == N-3 && i1 < N_bkup-3){
			j=2*N;
			k=j+1;
			
			coord1[0] = (x[j-2][0] + x[k-2][0]) / 2.0f;
			coord1[1] = (x[j-2][1] + x[k-2][1]) / 2.0f;
			coord1[2] = (x[j-2][2] + x[k-2][2]) / 2.0f;
			coord2[0] = (x[j-4][0] + x[k-4][0]) / 2.0f;
			coord2[1] = (x[j-4][1] + x[k-4][1]) / 2.0f;
			coord2[2] = (x[j-4][2] + x[k-4][2]) / 2.0f;
			del[0] = coord1[0] - coord2[0];
			del[1] = coord1[1] - coord2[1];
			del[2] = coord1[2] - coord2[2];
			rsq = sqrt(del[0]*del[0] + del[1]*del[1] + del[2]*del[2]);
			if(rsq < 0.001f)
				rsq = 0.001;
			x[j][0] = x[j - 2][0] + del[0] / rsq * R_INTRA;
			x[j][1] = x[j - 2][1] + del[1] / rsq * R_INTRA;
			x[j][2] = x[j - 2][2] + del[2] / rsq * R_INTRA;
			x[k][0] = x[k - 2][0] + del[0] / rsq * R_INTRA;
			x[k][1] = x[k - 2][1] + del[1] / rsq * R_INTRA;
			x[k][2] = x[k - 2][2] + del[2] / rsq * R_INTRA;
				
			N = i1 + 4;
			N2 = 2*N;
		}
		#endif
	#endif
	



	#pragma omp for	schedule(static)
	for (i=0; i<2*N; i++) 
	{
		// updating velocity by half a step
		// v = v(t+0.5dt)
		
		v[i][0] -= halfdtgamma*v[i][0];
		v[i][1] -= halfdtgamma*v[i][1];
		v[i][2] -= halfdtgamma*v[i][2];

		v[i][0] += halfdtMass*f[i][0];
		v[i][1] += halfdtMass*f[i][1];
		v[i][2] += halfdtMass*f[i][2];

		// updating position by a full step
		// x = x(t+dt)
		
		del[0] = v[i][0]*dt;
		del[1] = v[i][1]*dt;
		del[2] = v[i][2]*dt;
		
		#ifdef PULL
			if(i < 2){
				// only update the z coordinate for the first bp
				x[i][2]+= del[2] ;
				xRef[i][2] += del[2] ;
				continue;
			}
		#endif

		x[i][0]+= del[0] ;
		x[i][1]+= del[1] ;
		#ifndef TWOD
		x[i][2]+= del[2] ;
		#endif

		// updating displacement since last neighbour rebuild
		
		xRef[i][0] += del[0] ;
		xRef[i][1] += del[1] ;
		#ifndef TWOD
		xRef[i][2] += del[2] ;
		#endif

	}

	
	// calculate forces for the next timestep
	// f = f(t+dt)
	
/********************* BEGIN FORCE CALCULATION ***********************/

	// Initialize with random forces instead of zeros 
	
	#pragma omp for	schedule(static)
	for (i=0; i<2*N; i++) {
		#ifdef _OPENMP
		f[i][0]=ziggurat(thread_num)*randFmult;
		f[i][1]=ziggurat(thread_num)*randFmult;
		f[i][2]=ziggurat(thread_num)*randFmult;
		#else
		//f[i][0] = r4_nor (&seed , kn, fn, wn )*randFmult;
		//f[i][1] = r4_nor (&seed , kn, fn, wn )*randFmult;
		//f[i][2] = r4_nor (&seed , kn, fn, wn )*randFmult;
		f[i][0] = rng_gaussian_real()*randFmult;
		f[i][1] = rng_gaussian_real()*randFmult;
		f[i][2] = rng_gaussian_real()*randFmult;
		#endif

	}
	
	// adding the pulling force
	#ifdef PULL
		f[0][2] -= f_PULL;
		f[1][2] += f_PULL;
	#endif


//	float randF [2*N]; if (t>890 && t<910) getMagnitude(f, randF);

	// private force array
/*	
	float f[2*N][3];
	int myflag[2*N];
	for (i=0; i<2*N; i++) myflag[i]=0;
	zero(f);
*/
	// Calculate forces from intra-strand bonds

	#pragma omp for	schedule(static) reduction(+:intraE)

	#ifdef CIRCULAR
	for (i=0; i<2*N; i++)	// circular case is periodical
		intraE += harmonic(i, (i+2)%(2*N), K_INTRA, R_INTRA);
	#else
	for (i=0; i<2*N-2; i++) // linear case has 2 less bonds
		intraE += harmonic(i, i+2, K_INTRA, R_INTRA);
	#endif

//	float intraF [2*N];	if (t>890 && t<910) getMagnitude(f, intraF);

	// Calculate forces from inter-strand interaction
	
	// update the isbound information to find the junction, added on 24/10/2017
	for (i=N-1; i>=0; i--) {
		j=2*i;
		k=j+1;
		
		del[0] = x[j][0] - x[k][0];
		del[1] = x[j][1] - x[k][1];
		del[2] = x[j][2] - x[k][2];
		
		rsq = del[0]*del[0] + del[1]*del[1] + del[2]*del[2];
		
		if (rsq >= CUT_INTER*CUT_INTER)
			isBound[i] = 0;
		else
			isBound[i] = 1;
	}
	
	// find the junction
	if(isBound[Junc_index] == 1){
		for(ii = Junc_index - 1; ii >= 0; ii--){
			if(isBound[ii] == 1)
				Junc_index--;
			else{
				Junc_index = ii + 1;
				break;
			}
		}
	}
	if(isBound[Junc_index] == 0){
		for(ii = Junc_index + 1; ii <= N - N_bpfix; ii++){
			if(isBound[ii] == 0)
				Junc_index++;
			else{
				Junc_index = ii;
				break;
			}
		}
	}
	
	//i = Junc_index;	// temporarily store
	//x1 = x[2 * N - 2][0];
	//x2 = x[2 * N - 4][0];
	//// keep x1 smaller than x2
	//if(x1 > x2){
	//	for(ii = N - 2; ii > i - 1; ii--){
	//		if(x[2 * ii + 0][0] < x[2 * ii + 2][0])
	//			Junc_index = ii;
	//		else{
	//			break;
	//		}
	//	}
	//}
	//else{
	//	for(ii = N - 2; ii > i - 1; ii--){
	//		if(x[2 * ii + 0][0] > x[2 * ii + 2][0])
	//			Junc_index = ii;
	//		else{
	//			break;
	//		}
	//	}
	//}

	#pragma omp for reduction(+:interE,dihedralE) schedule(static)
	for (i=N-1; i>=0; i--) {
		j=2*i;
		k=j+1;
		
		#ifndef CIRCULAR	
		/* if ( i == 0 ) {  */
		/* 	harmonic(j, k, K_INTER, R_INTER); */
		/* 	angleE += angle( j+2, j, k, kMult, dkMult ); */
		/* 	angleE += angle( k+2, k, j, kMult, dkMult ); */
		/* 	continue; */
		/* } */
		
		//#if defined ALLFIX
		//	if ( i >= 0 ) {
		#if defined ETE
			if ( i >= Junc_index + Win_Size || i > N-N_bpfix-1 ) {
		//#elif defined ONEBPFIX
		//	if ( i > N-2 ) {
		#else
			if ( i > N-N_bpfix-1 ) {
		#endif
			harmonic(j, k, K_INTER, R_INTER);
			//angleE += angle( j-2, j, k, kMult, dkMult );
			//angleE += angle( k-2, k, j, kMult, dkMult );
			//continue;
		}
		#endif
		
		#ifdef LOWTRELAX
			//harmonic(2 * N - 2, 2 * N - 1, K_INTER, R_INTER);
			harmonic(2 * N - 4, 2 * N - 3, K_INTER, R_INTER);
			harmonic(2 * N - 6, 2 * N - 5, K_INTER, R_INTER);
			harmonic(2 * N - 8, 2 * N - 7, K_INTER, R_INTER);
			
			for (ii=0; ii<N; ii++) {
				j=2*ii;
				k=j+1;
				del[0] = x[j][0] - x[k][0];
				del[1] = x[j][1] - x[k][1];
				del[2] = x[j][2] - x[k][2];
				
				rsq = del[0]*del[0] + del[1]*del[1] + del[2]*del[2];
				
				if (rsq >= CUT_INTER*CUT_INTER)
					isBound[ii] = 0;
				else
					isBound[ii] = 1;
			}
			
			break;
		#endif
		
		// End To End folding restriction
		#ifdef ETE
		if(i < Junc_index - Win_Size)	
			continue;
		#endif
		
		// ignore the h-bond for the free bps
		if(i < N_bpfree)	// free several bp at the tail to accelerate the melting, to skip waiting
			continue;
		#ifdef FREEBOTHENDS
		if ( i > N-N_bpfree2-1)	// last bp can also be freed
			continue;
		#else
		if ( i > N-N_bpfree2-2 && i < N-1 )	// last bp linked like a bond
			continue;
		#endif

		// circular case is periodical 
		inter = harcos(j, k, &c, &s);


		if  ( inter != 0 ) 
		{ 
			interE += inter;
			isBound[i] = 1;

			kMult  = 0.5 * ( 1 + c ) ;
		    	dkMult = -0.5 * rk * s  ;
			
			#ifdef CIRCULAR
				dihedralE += dihedral ((j+N2-2)%N2, j, k, (k+2)%N2, sin1, cos1, kMult, dkMult);
				dihedralE += dihedral ((j+2)%N2, j, k, (k+N2-2)%N2, sin2, cos2, kMult, dkMult);
				angleE += angle( (j+N2-2)%N2, j, k, kMult, dkMult );
				angleE += angle( (k+N2-2)%N2, k, j, kMult, dkMult );
				angleE += angle( (j+2)%N2   , j, k, kMult, dkMult );
				angleE += angle( (k+2)%N2   , k, j, kMult, dkMult );
			
			#else
				if(i > 0){
					angleE += angle( (j-2), j, k, kMult, dkMult );
					angleE += angle( (k-2), k, j, kMult, dkMult );
				}
				if(i < N-1){
					angleE += angle( (j+2), j, k, kMult, dkMult );
					angleE += angle( (k+2), k, j, kMult, dkMult );
				}
				if(i > 0 && i < N-1){
					dihedralE += dihedral ((j-2), j, k, (k+2), sin1, cos1, kMult, dkMult);
					dihedralE += dihedral ((j+2), j, k, (k-2), sin2, cos2, kMult, dkMult);
				}
			#endif
			
			
			//if (i>0) {
			//  dihedralE += dihedral ((j+N2-2)%N2, j, k, (k+2)%N2, sin1, cos1, kMult, dkMult);
			//  dihedralE += dihedral ((j+2)%N2, j, k, (k+N2-2)%N2, sin2, cos2, kMult, dkMult);
			//	
			//  angleE += angle( (j+N2-2)%N2, j, k, kMult, dkMult );
			//  angleE += angle( (k+N2-2)%N2, k, j, kMult, dkMult );
			//}
			//angleE += angle( (j+2)%N2   , j, k, kMult, dkMult );
			//angleE += angle( (k+2)%N2   , k, j, kMult, dkMult );
		}

		else	isBound[i]=0;
	}
	
	#ifdef BARWIND
	for (i=0; i<N; i++) {
		j=2*i;
		k=j+1;
		
		//// modified on Oct/20/2017
		//ete_J = 0;	// used here to determine whether the bp is in the main duplex
		//for(ii = i; ii < N-6; ii++){
		//	if(isBound[i + 1] == 0)
		//		ete_J = 1;	
		//	if(isBound[ii + 1] == 0 && isBound[ii + 2] == 0)
		//		ete_J = 1;	
		//}
		//// end of modification
		//
		//if(isBound[i] == 1 && ete_J == 0){
		if(isBound[i] == 1){
			// atom j
			del[1] = x[j][1]; 
			del[2] = x[j][2]; 
			//current bond length
			r = sqrt(del[1] * del[1] + del[2] * del[2]);
			dr = r - R_INTER / 2.0;
			kdr = 50.0 * dr;
			fmult = -2.0 * kdr / r;
		
			del[1] *= fmult;
			del[2] *= fmult;
			f[j][1] += del[1];
			f[j][2] += del[2];
			
			// atom k
			del[1] = x[k][1]; 
			del[2] = x[k][2]; 
			//current bond length
			r = sqrt(del[1] * del[1] + del[2] * del[2]);
			dr = r - R_INTER / 2.0;
			kdr = 50.0 * dr;
			fmult = -2.0 * kdr / r;
		
			del[1] *= fmult;
			del[2] *= fmult;
			f[k][1] += del[1];
			f[k][2] += del[2];
		}
	}
		// add tube repulsion to single strand section in ladder
		//#ifdef LADDER
		//x1 = x[2 * Junc_index][0];
		//x2 = x[2 * N - 2][0];
		//// keep x1 smaller than x2
		//if(x1 > x2){
		//	x0 = x1;
		//	x1 = x2;
		//	x2 = x0;
		//}
		//
		//for (i=0; i<Junc_index; i++){
		//	j=2*i;
		//	k=j+1;
		//	
		//	// atom j
		//	if(x[j][0] > x1 && x[j][0] < x2){
		//		del[1] = x[j][1]; 
		//		del[2] = x[j][2]; 
		//		//current bond length
		//		r = sqrt(del[1] * del[1] + del[2] * del[2]);
		//		if(r < 0.8){
		//			dr = r - 0.8;
		//			kdr = 500.0 * dr;
		//			fmult = -2.0 * kdr / r;
		//			
		//			del[1] *= fmult;
		//			del[2] *= fmult;
		//			f[j][1] += del[1];
		//			f[j][2] += del[2];
		//		}
		//	}
		//	
		//	// atom k
		//	if(x[k][0] > x1 && x[k][0] < x2){
		//		del[1] = x[k][1]; 
		//		del[2] = x[k][2]; 
		//		//current bond length
		//		r = sqrt(del[1] * del[1] + del[2] * del[2]);
		//		if(r < 0.8){
		//			dr = r - 0.8;
		//			kdr = 500.0 * dr;
		//			fmult = -2.0 * kdr / r;
		//			
		//			del[1] *= fmult;
		//			del[2] *= fmult;
		//			f[k][1] += del[1];
		//			f[k][2] += del[2];
		//		}
		//	}
		//}
		//#endif
	#endif
	
	// confine helix between two parallel walls (z=-0.55, z=0.55)
	#ifdef HTWOD
	for (j=0; j<N*2; j++) {
		if(x[j][2] > R_INTER / 2.0){
			dr = x[j][2] - R_INTER / 2.0;
			f[j][2] += -100.0 * dr;
		}
		if(x[j][2] < -R_INTER / 2.0){
			dr = -R_INTER / 2.0 - x[j][2];
			f[j][2] += 100.0 * dr;
		}
	}
	#endif


//	float interF[2*N]; if (t>890 && t<910) getMagnitude(f, interF);

	// Calculate forces form hard-core repulsion
	
	#pragma omp for	reduction(+:hardE) schedule(static)
	for (i=0; i<2*N; i++) 
	{
		for (k=1; k<neigh[i][0]+1; k++) 
		{
		j = neigh[i][k];
		
		#ifdef INVGAUSSIAN	// only applying Gaussian model to the duplex part, single strand will also be excluded from the duplex
		if(i/2 >= Junc_index && j/2 >= Junc_index){
			continue;
		}
		#endif
		#ifdef INVSECGAUSSIAN	// second version: only applying Gaussian model to the duplex part, single strand won't be excluded from the duplex
		if(i/2 >= Junc_index || j/2 >= Junc_index){
			continue;
		}
		#endif
		// #ifdef GAUSSIAN	// only applying Gaussian model to the single strand
		// if(isBound[i/2] == 0 && isBound[j/2] == 0){
			// continue;
		// }
		// #endif
		#ifdef GAUSSIAN	// only applying Gaussian model to the single strand, single strand will also be excluded from the duplex
		if(i/2 < Junc_index && j/2 < Junc_index){
			continue;
		}
		#endif
		#ifdef SECGAUSSIAN	// second version: only applying Gaussian model to the single strand, single strand won't be excluded from the duplex
		if(i/2 < Junc_index || j/2 < Junc_index){
			continue;
		}
		#endif
		#ifdef PUREGAUSSIAN
			break;
		#endif

		// intra strand hardcore repulsion
		if ( (i+j)%2 == 0 ) 	
		//	intraHardE += hardcore_4_2 (i, j, sigma2_intra);
			intraHardE += softcore (i, j, R_HC_INTRA);

		// inter strand hardcore repulsion
		else			
		//	interHardE += hardcore_4_2 (i, j, sigma2_inter);
			interHardE += softcore (i, j, R_HC_INTER);

		}
	}
/*
	if (t>890 && t<910) {
		float nonbF[2*N];
		getMagnitude(f, nonbF);
		float forces[2*N][4];
		float max[4]={0,0,0,0};

		printf("step %d\nrand\tintra\tinter\tnonbond\n", t);
		for (i=0; i<2*N; i++) {
			forces[i][3]=abs(nonbF[i]-interF[i]);
			forces[i][2]=abs(interF[i]-intraF[i]);
			forces[i][1]=abs(intraF[i]-randF[i]);
			forces[i][0]=randF[i];
			for (j=0; j<4; j++) if (forces[i][j]>max[j]) max[j]=forces[i][j];
			
			printf("%d\t%f\t%f\t%f\t%f\n",i, forces[i][0], forces[i][1], forces[i][2], forces[i][3]);
		}
		printf("MAX\n%f\t%f\t%f\t%f\n", max[0], max[1], max[2], max[3]);

	}
*/


/****************** END OF FORCE CALCULATION **************************/
	
	
	// final update on velocity 
	// v = v(t+dt)
	#pragma omp for schedule(static)	
	for (i=0; i<2*N; i++)  { 

			v[i][0] += f[i][0]*halfdtMass ;
			v[i][1] += f[i][1]*halfdtMass ;
			v[i][2] += f[i][2]*halfdtMass ;
			
			v[i][0] *=halfdtgammanorm;
                        v[i][1] *=halfdtgammanorm;
	                v[i][2] *=halfdtgammanorm;
	}
	}
}
