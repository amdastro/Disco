
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int countlines(char * filename){
   FILE *pFile = fopen(filename, "r");
   int lines=0;
   char c;
   while ((c = fgetc(pFile)) != EOF){
      if (c == '\n') ++lines;
   }
   fclose(pFile);
   return(lines);
}

int main( int argc , char * argv[] ){

   double sigma = 10.*2.*M_PI;
   int N2 = 1000;

   if( argc<2 ){
      printf("Please input filename.\n");
      return(1);
   }

   char filename[256];
   sprintf(filename,"%s",argv[1]);
   
   int Nl = countlines( filename );
   printf("#File = %s has %d lines\n",filename,Nl);

   double * x  = (double *) malloc( Nl*sizeof(double) );
   double * y  = (double *) malloc( Nl*sizeof(double) );
   double * y2 = (double *) malloc( Nl*sizeof(double) );

   FILE * pFile = fopen( filename , "r" );
   int i;
   for( i=0 ; i<Nl ; ++i ){
      fscanf( pFile , "%lf %lf %lf %*e %*e %*e %*e %*e %*e %*e %*e %*e %*e %*e %*e %*e\n", x+i,y+i,y2+i);
      //y[i] /= r*r*r;//sqrt(r);
   }
   fclose( pFile );

   printf("# xmax = %e\n",x[Nl-1]);
   double xnew[N2];
   double ynew[N2];
   double y2new[N2];

   int j;
   for( j=0 ; j<N2 ; ++j ){
      xnew[j]  = ((double)j+.5)*x[Nl-1]/(double)N2;
      ynew[j]  = 0.0;
      y2new[j] = 0.0;
   }
   for( i=1 ; i<Nl-1 ; ++i ){
      double dx = .5*(x[i+1]-x[i-1]);
      for( j=0 ; j<N2 ; ++j ){
         double xi = x[i];
         double xj = xnew[j];
         double xx = xi-xj;
         double f = exp(-.5*xx*xx/sigma/sigma)/sqrt(2.*M_PI)/sigma;
         ynew[j]  += f*y[i]*dx;
         y2new[j] += f*y2[i]*dx;
      }
   }

   for( j=0 ; j<N2 ; ++j ){
      printf("%e %e %e\n",xnew[j],ynew[j],y2new[j]);
   }

   free(x);
   free(y);
   free(y2);

   return(0);
}
