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
	/*dichiarazioni variabili*/
    int rank,world_size,tag;
	int n,nloc,i,p,r;
	int *potenze,passi=0;
	double *vett_loc, *vett = NULL; // Inizializzato a NULL
	double sommaloc=0, tmp;
	double T_inizio,T_fine,T_max;

	MPI_Status info;
	
	/*Inizializzazione dell'ambiente di calcolo MPI*/
	MPI_Init(&argc,&argv);
	/*assegnazione IdProcessore a rank*/
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	/*assegna numero processori a world_size*/
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	/* lettura e inserimento dati*/
	if (rank==0)
	{
		printf("Inserire il numero di elementi da sommare: \n");
		fflush(stdout);
		scanf("%d",&n);
		
        // MPI_Scatter richiede che ogni processo riceva lo stesso numero di elementi.
        if (n % world_size != 0) {
            fprintf(stderr, "Errore: Il numero di elementi (%d) deve essere un multiplo del numero di processi (%d) per usare MPI_Scatter.\n", n, world_size);
            // Termina tutti i processi MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

       	vett=(double*)calloc(n,sizeof(double));
        
        srand((double) time(0)); 
        for(i=0; i<n; i++)
		{
			*(vett+i)=rand()%5-2;
		}
		
		if (n<100)
		{
			for (i=0; i<n; i++)
			{
				printf("\nElemento %d del vettore = %f",i,*(vett+i));
			}
            printf("\n\n");
        }
	}

	/*invio del valore di n a tutti i processori*/
	MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
	
    // Poiché n è divisibile per world_size, tutti i processi ricevono lo stesso numero di elementi.
	nloc = n / world_size;
	
    /*allocazione di memoria del vettore per le somme parziali */
	vett_loc=(double*)calloc(nloc, sizeof(double));

    
    // Distribuisce nloc elementi dal vettore 'vett' (su P0) a tutti i processi,
    // salvandoli nel rispettivo 'vett_loc'.
    MPI_Scatter(vett,           // buffer di invio (significativo solo su root 0)
                nloc,           // numero di elementi da inviare A CIASCUN processo
                MPI_DOUBLE,     // tipo di dato da inviare
                vett_loc,       // buffer di ricezione
                nloc,           // numero di elementi da ricevere
                MPI_DOUBLE,     // tipo di dato da ricevere
                0,              // id del processo root (che invia)
                MPI_COMM_WORLD);// comunicatore

    // Il processo P0 può ora liberare la memoria del vettore completo
    if (rank == 0) {
        free(vett);
    }
	
	/* sincronizzazione dei processori*/
	MPI_Barrier(MPI_COMM_WORLD);
 
	T_inizio=MPI_Wtime();

	for(i=0;i<nloc;i++)
	{
		/*ogni processore effettua la somma parziale*/
		sommaloc=sommaloc+*(vett_loc+i);
	}

	//  calcolo di p=log_2 (world_size)
	p=world_size;
	while(p!=1)
	{
		p=p>>1;
		passi++;
	}
 
	/* creazione del vettore potenze, che contiene le potenze di 2*/
	potenze=(int*)calloc(passi+1,sizeof(int));
	for(i=0;i<=passi;i++)
	{
		potenze[i]=p<<i;
	}

	/* calcolo delle somme parziali e combinazione dei risultati parziali */
	for(i=0;i<passi;i++)
	{
		int partner = rank ^ potenze[i];
		// Tutti i processi eseguono la stessa chiamata.
		MPI_Sendrecv(&sommaloc, 1, MPI_DOUBLE, partner, 0,  // Parte di invio
					&tmp,      1, MPI_DOUBLE, partner, 0,  // Parte di ricezione
					MPI_COMM_WORLD, &info);

		sommaloc += tmp;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	T_fine=MPI_Wtime()-T_inizio;
 
	/* calcolo del tempo totale di esecuzione*/
	MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

	/*stampa a video dei risultati finali*/
	if(rank==0)
	{
		printf("\nprocessori impegnati: %d\n", world_size);
		printf("\nLa somma e': %f\n", sommaloc);
		printf("\nTempo calcolo locale (su P0): %lf\n", T_fine);
		printf("\nMPI_Reduce max time: %f\n",T_max);
	}
	//printf("\n Processore %d, somma locale = %.3f\n", rank, sommaloc);
 
    free(vett_loc);
    free(potenze);

	/*routine chiusura ambiente MPI*/
	MPI_Finalize();

    return 0;
}