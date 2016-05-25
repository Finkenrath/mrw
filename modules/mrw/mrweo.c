
/*******************************************************************************
*
* File mrweo.c
*
* Copyright (C) 2012, 2013 Martin Luescher, 2013 Bjoern Leder, Jacob Finkenrath
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Mass and twisted-mass reweighting factors
*
* The externally accessible functions are
*
*   complex_dble mrw1eo(mrw_masses_t ms,int tm,int isp,double *sqnp,double *sqne,
*                    int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqne, solves the twisted-mass Dirac equation
*     using the mass/twisted-mass parameters in ms, sets sqnp to the square norm
*     of the solution and returns -ln(w^(1)) (see the documentation).
*     The solver is specified by the parameter set number isp.
*     The argument status must be pointing to an array of at least 1,1
*     and 3 elements, respectively, in the case of the CGNE, SAP_GCR and
*     DFL_SAP_GCR solver. On exit the array elements return the status
*     values reported by the solver program.
*
*  complex_dble mrw2eo(mrw_masses_t ms,int tm,int *isp,complex_dble *lnw1,
*                   double *sqnp,double *sqne,int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqne, solves the twisted-mass Dirac equation
*     twice using the mass/twisted-mass parameters in ms, sets sqnp[0,1] to the
*     square norm of the solutions, sets lnw1[0,1] to -ln(w^(1)) and returns
*     -ln(w^(2)) (see the documentation).
*     The solvers for the two solves are specified by the parameter set numbers
*     isp[0,1].
*     The argument status must be pointing to an array of at least 2,2
*     and 6 elements, respectively, in the case of the CGNE, SAP_GCR and
*     DFL_SAP_GCR solver. On exit the array elements return the status
*     values reported by the solver program for first (first
*     half of the array) and second solve (second half).
*
* double mrw3eo(mrw_masses_t ms,int *isp,complex_dble *lnw1,double *sqnp,
*             double *sqne,int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqne, solves the twisted-mass Dirac equation
*     tree times using the mass/twisted-mass parameters in ms, sets sqnp[0,1] to the
*     square norm of the first two solutions, sets lnw1[0,1] to -ln(w^(1)) and returns
*     -ln(w^(4,tm)) (see the documentation).
*     The solvers for the two solves are specified by the parameter set numbers
*     isp[0,1].
*     The argument status must be pointing to an array of at least 4,4
*     and 9 elements, respectively, in the case of the CGNE, SAP_GCR and
*     DFL_SAP_GCR solver. On exit the array elements return the status
*     values reported by the solver program for first (first
*     third of the array), second solve (second third) and third solve
*     (last third).
*     If ms.d1==-ms.d2 and ms.m1==ms.m2 the second solve is skipped and
*     sqnp[1], lnw1[1] and status[3-5] are set to zero.
* 
* 
* Notes:
* 
* See doc/mrw.pdf for more details.
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define MRWEO_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "sflds.h"
#include "linalg.h"
#include "sap.h"
#include "dfl.h"
#include "forces.h"
#include "mrw.h"
#include "global.h"
#include "dirac.h"
#include "sw_term.h"

#define N0 (NPROC0*L0)
#define MAX_LEVELS 8
#define BLK_LENGTH 8

static int cntr[MAX_LEVELS];
static double smxr[MAX_LEVELS];
static int cnti[MAX_LEVELS];
static double smxi[MAX_LEVELS];

static double set_eta(spinor_dble *eta)
{
   random_sd(VOLUME/2,eta,1.0);
   set_sd2zero(VOLUME/2,eta+(VOLUME/2));
   bnd_sd2zero(EVEN_PTS,eta);
   
   return norm_square_dble(VOLUME/2,1,eta);
}

static double get_kappa(double m)
{
 return 1.0/(2.0*(4.0+m));
}

static void get_psi(double m,double mu,int ihc,spinor_dble *eta,
                    spinor_dble *psi,int isp, int* stat)
{
   double kappa;
   spinor_dble **wsd;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;
   
   stat[0]=0;
   stat[1]=0;
   stat[2]=0;
   
   sp=solver_parms(isp);
   
   if (sp.solver==CGNE)
   {
      wsd=reserve_wsd(1);
      if (ihc==0)
      {
         mu*=-1.0;
         mulg5_dble(VOLUME/2,eta);
	 tm=tm_parms();
	 if (tm.eoflg==1)
	 {
	    if (mu>0.0)
	       set_tm_parms(tm.eoflg,fabs(tm.mu));
	    else
	       set_tm_parms(tm.eoflg,-fabs(tm.mu));
	 }
      }
   }
   else
   {   
      if (ihc==1)
      {
         mu*=-1.0;
         mulg5_dble(VOLUME/2,eta);
	 	 
	 tm=tm_parms();
	 if (tm.eoflg==1)
	 {
	    if (mu>0.0)
	       set_tm_parms(tm.eoflg,fabs(tm.mu));
	    else
	       set_tm_parms(tm.eoflg,-fabs(tm.mu));
	 }
      }
   }

   set_sw_parms(m);

   if (sp.solver==CGNE)
   {
      tmcgeo(sp.nmx,sp.res,mu,eta,wsd[0],stat);
      error_root(stat[0]<0,1,"get_psi [mrweo.c]",
               "CGNE solver failed (mu = %.4e, parameter set no %d, "
               "status = %d)",mu,isp,stat[0]);     
      Dwhat_dble(mu,wsd[0],psi);      
   }
   else if (sp.solver==SAP_GCR)
   {
      kappa=get_kappa(m);
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy,kappa,sp.mu);

      sap_gcr(sp.nkv,sp.nmx,sp.res,mu,eta,psi,stat);
      error_root(stat[0]<0,1,"get_psi [mrweo.c]",
               "SAP_GCR solver failed (mu = %.4e, parameter set no %d, "
               "status = %d)",mu,isp,stat[0]);      
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      kappa=get_kappa(m);
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy,kappa,sp.mu);

      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.res,mu,eta,psi,stat);      
      error_root((stat[0]<0)||(stat[1]<0),1,
               "get_psi [mrweo.c]","DFL_SAP_GCR solver failed "
               "(mu = %.4e, parameter set no %d, status = (%d,%d,%d))",
               mu,isp,stat[0],stat[1],stat[2]);
   }
   else
      error_root(1,1,"get_psi [mrweo.c]","Unknown solver");

   set_sd2zero(VOLUME/2,psi+(VOLUME/2));

   if (sp.solver==CGNE)
   {
      release_wsd();
      if (ihc==0)
      {
         mulg5_dble(VOLUME/2,psi);
         mulg5_dble(VOLUME/2,eta);
      }
   }
   else
   {   
      if (ihc==1)
      {
         mulg5_dble(VOLUME/2,psi);
         mulg5_dble(VOLUME/2,eta);
      }
   }
}


complex_dble mrw1eo(mrw_masses_t ms,int tm,int isp,double *sqnp,double *sqne,int *status)
{
   complex_dble lnw,z;
   spinor_dble *eta,*psi1,**wsd;
   /*tm_parms_t tmp;

   error_root(tm==0,1,"mrw1eo [mrweo.c]",
            "Even-odd mass reweighting not supported yet.");     
   
   tmp=tm_parms();
   if (tmp.eoflg!=1)
      set_tm_parms(1);*/
   set_tm_parms(1,ms.mu_odd0);
   
   lnw.re=0.0;
   lnw.im=0.0;
   (*sqne)=0.0;
   (*sqnp)=0.0;
   status[0]=0;
   status[1]=0;
   status[2]=0;
   
   if (ms.d1==0.0)
      return lnw;
   
   wsd=reserve_wsd(2);
   
   eta=wsd[0];
   psi1=wsd[1];
   (*sqne)=set_eta(eta);
   
   get_psi(ms.m1,ms.mu1,1,eta,psi1,isp,status);
   
   if (tm)
   {
      z=spinor_prod5_dble(VOLUME/2,1,psi1,eta);
      lnw.re=-ms.d1*z.im;
      lnw.im=ms.d1*z.re;      
   }
   else
   {
      z=spinor_prod_dble(VOLUME/2,1,psi1,eta);
      lnw.re=ms.d1*z.re;
      lnw.im=ms.d1*z.im;
   }
      
   (*sqnp)=norm_square_dble(VOLUME/2,1,psi1);
   
   release_wsd();

   return lnw;
}


complex_dble mrw2eo(mrw_masses_t ms,int tm,int *isp,complex_dble *lnw1,
                  double *sqnp,double *sqne,int *status)
{
   complex_dble lnw,z;
   spinor_dble *eta,*psi1,*psi2,**wsd;
   /*tm_parms_t tmp;

   error_root(tm==0,1,"mrw1eo [mrweo.c]",
            "Even-odd mass reweighting not supported yet.");     

   tmp=tm_parms();
   if (tmp.eoflg!=1)
      set_tm_parms(1);*/
   set_tm_parms(1,ms.mu_odd0);
   
   
   lnw.re=0.0;
   lnw.im=0.0;
   (*sqne)=0.0;
   
   if ((ms.d1==0.0) && (ms.d2==0.0))
      return lnw;
   
   wsd=reserve_wsd(3);
   
   psi1=wsd[0];
   psi2=wsd[1];
   eta=wsd[2];
   (*sqne)=set_eta(eta);
      
   get_psi(ms.m1,ms.mu1,1,eta,psi1,isp[0],status);   
   get_psi(ms.m2,ms.mu2,tm,eta,psi2,isp[1],status+3);

   if (tm)
   {
      z=spinor_prod5_dble(VOLUME/2,1,psi1,eta);
      lnw1[0].re=-ms.d1*z.im;
      lnw1[0].im=ms.d1*z.re;
      z=spinor_prod5_dble(VOLUME/2,1,psi2,eta);
      lnw1[1].re=-ms.d2*z.im;
      lnw1[1].im=ms.d2*z.re;
   }
   else
   {
      z=spinor_prod_dble(VOLUME/2,1,psi1,eta);
      lnw1[0].re=ms.d1*z.re;
      lnw1[0].im=ms.d1*z.im;
      z=spinor_prod_dble(VOLUME/2,1,psi2,eta);
      lnw1[1].re=ms.d2*z.re;
      lnw1[1].im=ms.d2*z.im;
   }

   z=spinor_prod_dble(VOLUME/2,1,psi1,psi2);
   lnw.re=lnw1[0].re+lnw1[1].re+ms.d1*ms.d2*z.re;
   lnw.im=lnw1[0].im-lnw1[1].im+ms.d1*ms.d2*z.im;

   if (tm==0)
      lnw1[1].im*=-1.0;
   
   sqnp[0]=norm_square_dble(VOLUME/2,1,psi1);
   sqnp[1]=norm_square_dble(VOLUME/2,1,psi2);

   release_wsd();

   return lnw;
}


double mrw3eo(mrw_masses_t ms,int *isp,complex_dble *lnw1,
                  double *sqnp,double *sqne,int *status)
{
   double d1,d2,lnw;
   complex_dble z;
   spinor_dble *eta,*psi1,*psi2,**wsd;
   /*tm_parms_t tm;

   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);*/
   set_tm_parms(1,ms.mu_odd0);
   
   lnw=0.0;
   (*sqne)=0.0;
   
   if ((ms.d1==0.0) && (ms.d2==0.0))
      return lnw;
   
   d1=-ms.mu1+sqrt(ms.mu1*ms.mu1+ms.d1);
   d2=-ms.mu2+sqrt(ms.mu2*ms.mu2+ms.d2);;

   wsd=reserve_wsd(3);
   
   psi1=wsd[0];
   psi2=wsd[1];
   eta=wsd[2];
   (*sqne)=set_eta(eta);
   
   get_psi(ms.m1,ms.mu1,1,eta,psi1,isp[0],status);

   z=spinor_prod5_dble(VOLUME/2,1,psi1,eta);
   lnw1[0].re=-d1*z.im;
   lnw1[0].im=d1*z.re;
   sqnp[0]=norm_square_dble(VOLUME/2,1,psi1);
   
   if ((ms.d2==-ms.d1)&&(ms.m1==ms.m2))
   {
      lnw1[1].re=0.0;
      lnw1[1].im=0.0;
      sqnp[1]=0.0;
      status[3]=0;
      status[4]=0;
      status[5]=0;
   }
   else
   {
      
      get_psi(ms.m2,ms.mu2,1,eta,psi2,isp[1],status+3);

      z=spinor_prod5_dble(VOLUME/2,1,psi2,eta);
      lnw1[1].re=-d2*z.im;
      lnw1[1].im=d2*z.re;
      sqnp[1]=norm_square_dble(VOLUME/2,1,psi2);
   
      if (ms.d2==-ms.d1)
      {
         assign_sd2sd(VOLUME,psi1,eta);
         mulr_spinor_add_dble(VOLUME/2,eta,psi2,-1.0);
         mulr_spinor_add_dble(VOLUME/2,psi2,psi1,1.0);
         z=spinor_prod_dble(VOLUME/2,1,eta,psi2);
      }
   }

   get_psi(ms.m2,ms.mu2,0,psi1,psi2,isp[1],status+6);

   lnw=norm_square_dble(VOLUME/2,1,psi2);
   
   if ((ms.d2==-ms.d1)&&(ms.m1==ms.m2))
      lnw*=(ms.d1*(ms.mu2*ms.mu2-ms.mu1*ms.mu1)-ms.d1*ms.d1);
   else
   {      
      if (ms.d2==-ms.d1)
         lnw=ms.d1*(z.re-ms.d1*lnw);
      else
         lnw=ms.d1*sqnp[0]+ms.d2*sqnp[1]+ms.d1*ms.d2*lnw;
   }

   release_wsd();

   return lnw;
}


static complex_dble sdet(void)
{
   int bc,ix,iy,t,n,ie;
   double c,pr,pi,mu;
   complex_dble z,lz;
   pauli_dble *m;
   sw_parms_t swp;
   tm_parms_t tm;
   
   tm=tm_parms();
   swp=sw_parms();
   mu=tm.mu_sdet;
   
   if ((4.0+swp.m0)>1.0)
      c=pow(4.0+swp.m0,-6.0);
   else
      c=1.0;

   for (n=0;n<MAX_LEVELS;n++)
   {
      cntr[n]=0;
      smxr[n]=0.0;
      
      cnti[n]=0;
      smxi[n]=0.0;
   }

   sw_term(NO_PTS);
   m=swdfld()+VOLUME;
   bc=bc_type();
   ix=(VOLUME/2);
   ie=0;

   while (ix<VOLUME)
   {
      pr=1.0;
      pi=1.0;
      iy=ix+BLK_LENGTH;
      if (iy>VOLUME)
         iy=VOLUME;

      for (;ix<iy;ix++)
      {
         t=global_time(ix);

         if (((t>0)||(bc==3))&&((t<(N0-1))||(bc!=0)))
         {
            z=det_pauli_dble(mu,m);
            pr=(c*z.re)*pr-(c*z.im)*pi;
	    pi=(c*z.im)*pr+(c*z.re)*pi;
	    
	    z=det_pauli_dble(mu,m+1);
	    pr=(c*z.re)*pr-(c*z.im)*pi;
	    pi=(c*z.im)*pr+(c*z.re)*pi;
         }

         m+=2;
      }

      if (pr!=0.0)
      {
         cntr[0]+=1;
         smxr[0]-=log(pr);

         for (n=1;(cntr[n-1]>=BLK_LENGTH)&&(n<MAX_LEVELS);n++)
         {
            cntr[n]+=1;
            smxr[n]+=smxr[n-1];

            cntr[n-1]=0;
            smxr[n-1]=0.0;
         }
      }
      else
         ie=1;
      
      if (pi!=0.0)
      {
         cnti[0]+=1;
         smxi[0]-=log(pi);

         for (n=1;(cnti[n-1]>=BLK_LENGTH)&&(n<MAX_LEVELS);n++)
         {
            cnti[n]+=1;
            smxi[n]+=smxi[n-1];

            cnti[n-1]=0;
            smxi[n-1]=0.0;
         }
      }
      else
         ie=1;
      
   }

   error(ie!=0,1,"sdet [mrweo.c]",
         "SW term has vanishing determinant");

   for (n=1;n<MAX_LEVELS;n++)
   {
      smxr[0]+=smxr[n];
      smxi[0]+=smxi[n];
   }

   lz.re=smxr[0];
   lz.im=smxi[0];
   
   return lz;
}

complex_dble get_cswdet(mrw_masses_t ms)
{
   complex_dble adet,ldet;
   
   set_tmsdet_parms(ms.mu_odd0);
   set_sw_parms(ms.m1);
   
   ldet=sdet();
 
   MPI_Reduce((double*) &ldet,(double*) &adet,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   MPI_Bcast((double*) &adet,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   
   return adet;
}
