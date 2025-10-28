#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/*--------------------------------------------------------------------------
  Inclusione del file che contiene le definizioni necessarie al preprocessore
  per l'utilizzo di MPI.
-----------------------------------------------------------------------------*/
#include <mpi.h>

int main (int argc, char **argv)
{	
    int rank,world_size;

	MPI_Status info;
	/*Inizializzazione dell'ambiente di calcolo MPI*/
	MPI_Init(&argc,&argv);
	/*assegnazione IdProcessore a rank*/
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	/*assegna numero processori a world_size*/
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    MPI_Comm MPI_grid; /* Nome della topologia*/
    int dim=2; /* dimensione */
    int ndim[2], period[2], reorder; /*caratteristiche della griglia */
    int coordinate[2];

    ndim[0] = 3; //row
    ndim[1]= 3; //col

    period[0] = 0; period[1]=0; // no period

    reorder = 0; //no reorder
    MPI_Cart_create(MPI_COMM_WORLD, dim, ndim, period, reorder,
    &MPI_grid);

    int grid_id;
    MPI_Comm_rank(MPI_grid, &grid_id); /* id nella griglia */
    MPI_Cart_coords(MPI_grid, grid_id, dim, coordinate);

    int a;
    if(coordinate[0]==coordinate[1]){
        a=coordinate[0]*coordinate[0];
    }else{
        a=coordinate[0]+2*coordinate[1];
    }

    printf("Proc:%d; %d:%d ; a:%d\n",rank,coordinate[0],coordinate[1],a);
	MPI_Finalize();

    return 0;
}