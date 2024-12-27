// harmonic cosine potential to represent breakable inter-strand bonds
// harmonic until r0        		v = k (r-r0)^2 - E_INTER
// scaled cosine btw r0 and rCut	v = 
// 0 after rCut				v = 0

static float harcos(int i, int j, float *c, float *s ) {	
  	float delx,dely,delz,rsq;
  	
	float fmult,energy,r,dr,rk;

	delx = x[i][0] - x[j][0];
	dely = x[i][1] - x[j][1];
	delz = x[i][2] - x[j][2];
	
	rsq = delx*delx + dely*dely + delz*delz;

	//terminate early if beyond cutoff
	
	if (rsq >= CUT_INTER*CUT_INTER ) return 0.0;
	

	r = sqrt(rsq);
	dr = r - R_INTER;
	
	// this is the harmonic part
	if ( r < R_INTER) {  				
		rk = K_INTER * dr;
		fmult = -2.0*rk/r;
		#ifdef SEQ
			if(BaseType[i/2] == 'A')
				energy = rk*dr - E_INTER_A;
			if(BaseType[i/2] == 'C')
				energy = rk*dr - E_INTER_C;
			if(BaseType[i/2] == 'a')
				energy = rk*dr - E_INTER_a;
		#else
			energy = rk*dr - E_INTER;
		#endif
		*c = 1;
		*s = 0;
	}
	// this is the cosine part
	else { 		
		rk = M_PI / (CUT_INTER - R_INTER);
		*c = cos(dr*rk);
		*s = sin(dr*rk);
		
		#ifdef SEQ
			if(BaseType[i/2] == 'A'){
				energy = E_INTER_A * 0.5 * ( 1 - *c) - E_INTER_A;
				fmult = -E_INTER_A * 0.5 * rk * *s / r;
			}
			if(BaseType[i/2] == 'C'){
				energy = E_INTER_C * 0.5 * ( 1 - *c) - E_INTER_C;
				fmult = -E_INTER_C * 0.5 * rk * *s / r;
			}
			if(BaseType[i/2] == 'a'){
				energy = E_INTER_a * 0.5 * ( 1 - *c) - E_INTER_a;
				fmult = -E_INTER_a * 0.5 * rk * *s / r;
			}
		#else
			energy = E_INTER * 0.5 * ( 1 - *c) - E_INTER;
			fmult = -E_INTER * 0.5 * rk * *s / r;
		#endif
	}
	
	// apply forces
	 	
	delx *= fmult;
	dely *= fmult;
	delz *= fmult;
	
	#pragma omp atomic update
	f[i][0] += delx;
	#pragma omp atomic update
	f[i][1] += dely;
	#pragma omp atomic update
	f[i][2] += delz;
	  
	#pragma omp atomic update
	f[j][0] -= delx;
	#pragma omp atomic update
	f[j][1] -= dely;
	#pragma omp atomic update
	f[j][2] -= delz;
	
	return energy;
	
}
