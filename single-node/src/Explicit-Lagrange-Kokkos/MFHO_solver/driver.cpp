#include "include/mfho.hpp"

int main (int argc, char* argv[]) {

  // check to see of a mesh was supplied when running the code
  if (argc == 1) {
      std::cout << "\n\n**********************************\n\n";
      std::cout << " ERROR:\n";
      std::cout << " Please supply a mesh \n";
      std::cout << "   ./MFHO my_mesh.vtk \n\n";
      std::cout << "**********************************\n\n" << std::endl;
      std::exit(EXIT_FAILURE);
  } // end if


  MPI_Init (&argc, &argv);
  Kokkos::initialize (argc, argv);

  int my_rank = 0;
  int num_proc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0){
    printf("Matrix-Free Arbitrary-Order Lagrangian Simulation Started \n");
  }

  //initialize Trilinos communicator class
  comm = Tpetra::getDefaultComm();

  // only use VTKPn for now
  // readVTKPn(argv[1], mesh, node, elem, zone, mat_pt, corner, ref_elem, num_dims, rk_num_bins);

  MPI_Barrier(MPI_COMM_WORLD);
  Kokkos::finalize ();
  MPI_Finalize ();
}
