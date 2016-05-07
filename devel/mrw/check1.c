
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2013, 2015 Bjoern Leder, Jacob Finkenrath
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Direct check of mrw1
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "sflds.h"
#include "linalg.h"
#include "dirac.h"
#include "sap.h"
#include "dfl.h"
#include "forces.h"
#include "global.h"
#include "mrw.h"


#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)


static int my_rank;
static FILE *flog=NULL,*fin=NULL;

static void read_bc_parms(void)
{
   int bc;
   double cG,cG_prime,cF,cF_prime;
   double phi[2],phi_prime[2];

   if (my_rank==0)
   {
      find_section("Boundary conditions");
      read_line("type","%d",&bc);

      phi[0]=0.0;
      phi[1]=0.0;
      phi_prime[0]=0.0;
      phi_prime[1]=0.0;
      cG=1.0;
      cG_prime=1.0;
      cF=1.0;
      cF_prime=1.0;

      if (bc==1)
         read_dprms("phi",2,phi);

      if ((bc==1)||(bc==2))
         read_dprms("phi'",2,phi_prime);

      if (bc!=3)
      {
         read_line("cG","%lf",&cG);
         read_line("cF","%lf",&cF);
      }

      if (bc==2)
      {
         read_line("cG'","%lf",&cG_prime);
         read_line("cF'","%lf",&cF_prime);
      }
   }

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(phi,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(phi_prime,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cG,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cG_prime,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF_prime,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   set_bc_parms(bc,cG,cG_prime,cF,cF_prime,phi,phi_prime);

   print_bc_parms();
}

int main(int argc,char *argv[])
{
   int irw,isp,status[6],mnkv;
   int bs[4],Ns,nmx,nkv,nmr,ncy,ninv;
   double kappa,m0,dm,mu0,mu,res,mxres,mnres;
   double sqne0,sqne1,sqnp0,sqnp1,sqnr,rmx;
	double w0,w1,npl;
   complex_dble lnr0,lnr1,dr,drmx;
   spinor_dble **wsd;
   solver_parms_t sp;
   mrw_masses_t ms;
   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);
      fin=freopen("check1.in","r",stdin);
      
      printf("\n");
      printf("Direct check of mrw1\n");
      printf("----------------------\n\n");
      
      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   mnkv=0;
   
   mxres=0.0;
   mnres=1.0;
   for (isp=0;isp<3;isp++)
   {
      read_solver_parms(isp);
      sp=solver_parms(isp);

      if (sp.res>mxres)
         mxres=sp.res;
      
      if (sp.res<mnres)
         mnres=sp.res;
      
      if (sp.nkv>mnkv)
         mnkv=sp.nkv;
   }
   
   read_bc_parms();
   
   if (my_rank==0)
   {
      find_section("SAP");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   set_sap_parms(bs,0,1,1,0.0,0.0);

   if (my_rank==0)
   {
      find_section("Deflation subspace");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      read_line("Ns","%d",&Ns);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);   
   set_dfl_parms(bs,Ns);

   if (my_rank==0)
   {
      find_section("Deflation subspace generation");
      read_line("kappa","%lf",&kappa);
      read_line("mu","%lf",&mu);
      read_line("ninv","%d",&ninv);
      read_line("nmr","%d",&nmr);
      read_line("ncy","%d",&ncy);
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);   
   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);   
   MPI_Bcast(&ninv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncy,1,MPI_INT,0,MPI_COMM_WORLD);
   set_dfl_gen_parms(kappa,mu,ninv,nmr,ncy);
   
   if (my_rank==0)
   {
      find_section("Deflation projection");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);
      read_line("res","%lf",&res);
      fclose(fin);
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);     
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);  
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);      
   set_dfl_pro_parms(nkv,nmx,res);

   set_lat_parms(6.0,1.0,0,NULL,1.234);
	
   print_solver_parms(status,status+1);
   print_sap_parms(0);
   print_dfl_parms(0);
   
   start_ranlux(0,1245);
   geometry();

   mnkv=2*mnkv+2;
   if (mnkv<(Ns+2))
      mnkv=Ns+2;
   if (mnkv<5)
      mnkv=5;
   
   alloc_ws(mnkv);
   alloc_wsd(6);
   alloc_wv(2*nkv+2);
   alloc_wvd(4);
   drmx.re=0.0;
   drmx.im=0.0;
   rmx=0.0;
      
	npl=(double)(6*(N0-1))*(double)(N1*N2*N3);
   for (irw=0;irw<2;irw++)
   {
      dm=1.0e-2;

      for (isp=0;isp<3;isp++)
      {
         if (isp==0)
         {
            m0=1.0877;
            mu0=1.0;
         }
         else if (isp==1)
         {
            dm/=100.0;
            m0=0.0877;
            mu0=1.0;
         }
         else
         {
            dm/=100.0;
            m0=-0.0123;
            mu0=1.0;
         }
      
         random_ud();

         if (isp==2)
         {
            dfl_modes(status);
            error_root(status[0]<0,1,"main [check1.c]",
                        "dfl_modes failed");
         }      

         ms.m1=m0;
         ms.d1=dm;
         ms.mu1=mu0;

         if (irw==0)
         {
            start_ranlux(0,8910+isp);
            lnr0=mrw1(ms,0,isp,&sqnp0,&sqne0,status);

            wsd=reserve_wsd(3);
            start_ranlux(0,8910+isp);
            random_sd(VOLUME,wsd[0],1.0);
				
			  w0=plaq_wsum_dble(0)/npl;
				
			  MPI_Reduce(&w0,&w1,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			  MPI_Bcast(&w1,1,MPI_DOUBLE,0,MPI_COMM_WORLD); 
			  ranlxd(&w0,1);
			  
			  printf("Plaq %.16f and rand %.8f\n",w1,w0);
				
				
            bnd_sd2zero(ALL_PTS,wsd[0]);
            sqne1=norm_square_dble(VOLUME,1,wsd[0]);
            set_sw_parms(ms.m1);
            tmcg(10000,mnres/10.0,ms.mu1,wsd[0],wsd[1],status);
            error_root(status[0]<0,1,"main [check1.c]",
                        "tmcg failed (nmx=1000), status: %d",status[0]);
            Dw_dble(ms.mu1,wsd[1],wsd[2]);
            sqnp1=norm_square_dble(VOLUME,1,wsd[2]);
            lnr1=spinor_prod_dble(VOLUME,1,wsd[2],wsd[0]);
            mulg5_dble(VOLUME,wsd[2]);
            Dw_dble(-ms.mu1,wsd[2],wsd[1]);
            mulg5_dble(VOLUME,wsd[1]);
            mulr_spinor_add_dble(VOLUME,wsd[1],wsd[0],-1.0);
            sqnr=norm_square_dble(VOLUME,1,wsd[1]);
            release_wsd();

            dr.re=fabs(sqne1-sqne0);
            dr.re+=fabs(sqnp1-sqnp0);
            dr.re+=fabs(ms.d1*lnr1.re-lnr0.re);
            dr.im=fabs(ms.d1*lnr1.im-lnr0.im);
         }
         else
         {
            start_ranlux(0,8910+isp);
            lnr0=mrw1(ms,1,isp,&sqnp0,&sqne0,status);

            wsd=reserve_wsd(3);
            start_ranlux(0,8910+isp);
            random_sd(VOLUME,wsd[0],1.0);
            bnd_sd2zero(ALL_PTS,wsd[0]);
            sqne1=norm_square_dble(VOLUME,1,wsd[0]);
            set_sw_parms(ms.m1);
            tmcg(10000,mnres/10.0,ms.mu1,wsd[0],wsd[1],status);
            error_root(status[0]<0,1,"main [check1.c]",
                        "tmcg failed (nmx=1000), status: %d",status[0]);
            Dw_dble(ms.mu1,wsd[1],wsd[2]);
            sqnp1=norm_square_dble(VOLUME,1,wsd[2]);
            lnr1=spinor_prod5_dble(VOLUME,1,wsd[2],wsd[0]);
            mulg5_dble(VOLUME,wsd[2]);
            Dw_dble(-ms.mu1,wsd[2],wsd[1]);
            mulg5_dble(VOLUME,wsd[1]);
            mulr_spinor_add_dble(VOLUME,wsd[1],wsd[0],-1.0);
            sqnr=norm_square_dble(VOLUME,1,wsd[1]);
            release_wsd();

            dr.re=fabs(sqne1-sqne0);
            dr.re+=fabs(sqnp1-sqnp0);
            dr.re+=fabs(-ms.d1*lnr1.im-lnr0.re);
            dr.im=fabs(ms.d1*lnr1.re-lnr0.im);
         }
         
         if (dr.re>drmx.re)
            drmx.re=dr.re;
         if (dr.im>drmx.im)
            drmx.im=dr.im;
         if (sqnr>rmx)
            rmx=sqnr;

         if (my_rank==0)
         {
            if (irw==0)
               printf("tm=0: ");
            else
               printf("tm=1: ");
            
            if ((isp==0)||(isp==1))
               printf("status = %d\n",status[0]);
            else if (isp==2)
               printf("status = (%d,%d,%d)\n",
                        status[0],status[1],status[2]);

            printf("diff: %.1e + i%.1e\n\n",dr.re,dr.im);
         }      
      
         error_chk();
      }
   }
   
   if (my_rank==0)
   {
      printf("\n");
      printf("max diff: %.1e + i%.1e\n",drmx.re,drmx.im);
      mxres*=sqrt((double)(VOLUME*NPROC*24));
      if (rmx>mxres)
         mxres=rmx;
      printf("(should be smaller/equal than %.1e)\n\n",mxres);
      fclose(flog);
   }
   
   MPI_Finalize();    
   exit(0);
}
