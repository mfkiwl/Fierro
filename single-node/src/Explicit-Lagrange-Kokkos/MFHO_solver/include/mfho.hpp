#include "matar.h"
#include "mesh.h"
#include "node_combination.h"

#include <stdio.h>
#include <iostream>
#include <sys/stat.h>
#include <ctime>
#include <mpi.h>
#include <math.h>

#include <Kokkos_Core.hpp>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_VerboseObject.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include "Tpetra_Import.hpp"


long long int num_nodes, num_elem;

typedef Tpetra::Map<>::local_ordinal_type LO;
typedef Tpetra::Map<>::global_ordinal_type GO;

typedef Tpetra::CrsMatrix<real_t,LO,GO> MAT;
typedef const Tpetra::CrsMatrix<real_t,LO,GO> const_MAT;
typedef Tpetra::MultiVector<real_t,LO,GO> MV;
typedef Tpetra::MultiVector<GO,LO,GO> MCONN;

typedef Kokkos::ViewTraits<LO*, Kokkos::LayoutLeft, void, void>::size_type SizeType;
typedef Tpetra::Details::DefaultTypes::node_type node_type;
using traits = Kokkos::ViewTraits<LO*, Kokkos::LayoutLeft, void, void>;

using array_layout    = typename traits::array_layout;
using execution_space = typename traits::execution_space;
using device_type     = typename traits::device_type;
using memory_traits   = typename traits::memory_traits;
using global_size_t = Tpetra::global_size_t;

typedef Kokkos::View<real_t*, Kokkos::LayoutRight, device_type, memory_traits> values_array;
typedef Kokkos::View<GO*, array_layout, device_type, memory_traits> global_indices_array;
typedef Kokkos::View<LO*, array_layout, device_type, memory_traits> indices_array;
//typedef Kokkos::View<SizeType*, array_layout, device_type, memory_traits> row_pointers;
typedef MAT::local_graph_device_type::row_map_type::non_const_type row_pointers;
//typedef Kokkos::DualView<real_t**, Kokkos::LayoutLeft, device_type>::t_dev vec_array;
typedef MV::dual_view_type::t_dev vec_array;
typedef MV::dual_view_type::t_host host_vec_array;
typedef Kokkos::View<const real_t**, array_layout, HostSpace, memory_traits> const_host_vec_array;
typedef Kokkos::View<const real_t**, array_layout, device_type, memory_traits> const_vec_array;
typedef MV::dual_view_type dual_vec_array;
typedef MCONN::dual_view_type dual_elem_conn_array;
typedef MCONN::dual_view_type::t_host host_elem_conn_array;
typedef MCONN::dual_view_type::t_dev elem_conn_array;
typedef Kokkos::View<const GO**, array_layout, HostSpace, memory_traits> const_host_elem_conn_array;
typedef Kokkos::View<const GO**, array_layout, device_type, memory_traits> const_elem_conn_array;

Teuchos::RCP<const Teuchos::Comm<int> > comm;
Teuchos::RCP<Tpetra::Map<LO,GO,node_type> > map; //map of node indices
Teuchos::RCP<Tpetra::Map<LO,GO,node_type> > sorted_map; //sorted contiguous map of node indices
Teuchos::RCP<Tpetra::Map<LO,GO,node_type> > ghost_node_map; //map of node indices with ghosts on each rank
Teuchos::RCP<Tpetra::Map<LO,GO,node_type> > all_node_map; //map of node indices with ghosts on each rank
Teuchos::RCP<Tpetra::Map<LO,GO,node_type> > element_map; //non overlapping map of elements owned by each rank used in reduction ops
Teuchos::RCP<Tpetra::Map<LO,GO,node_type> > all_element_map; //overlapping map of elements connected to the local nodes in each rank
Teuchos::RCP<Tpetra::Map<LO,GO,node_type> > sorted_element_map; //sorted contiguous map of element indices owned by each rank used in parallel IO
Teuchos::RCP<Tpetra::Map<LO,GO,node_type> > local_dof_map; //map of local dofs (typically num_node_local*num_dim)
Teuchos::RCP<Tpetra::Map<LO,GO,node_type> > all_dof_map; //map of local and ghost dofs (typically num_node_all*num_dim)
Teuchos::RCP<MCONN> global_nodes_in_elem_distributed; //element to node connectivity table
Teuchos::RCP<MCONN> node_nconn_distributed; //how many elements a node is connected to
Teuchos::RCP<MV> node_coords_distributed;
Teuchos::RCP<MV> ghost_node_coords_distributed;
Teuchos::RCP<MV> initial_node_coords_distributed;
Teuchos::RCP<MV> all_initial_node_coords_distributed;
Teuchos::RCP<MV> all_node_coords_distributed;

//Distributions of data used to print
Teuchos::RCP<MV> collected_node_coords_distributed;
Teuchos::RCP<MCONN> collected_nodes_in_elem_distributed;
Teuchos::RCP<MV> sorted_node_coords_distributed;
Teuchos::RCP<MCONN> sorted_nodes_in_elem_distributed;

//Boundary Conditions Data
//CArray <Nodal_Combination> Patch_Nodes;
size_t nboundary_patches;
size_t num_boundary_conditions;
int current_bdy_id;
CArrayKokkos<Node_Combination, array_layout, HostSpace, memory_traits> Boundary_Patches;
std::map<Node_Combination,LO> boundary_patch_to_index; //maps patches to corresponding patch index (inverse of Boundary Patches array)

//file readin variables
std::ifstream *in = NULL;
std::streampos before_condition_header;
std::string filename;
int words_per_line, elem_words_per_line;
enum node_ordering_convention {IJK, ENSIGHT};
node_ordering_convention active_node_ordering_convention;

//file output variables
int file_index, nsteps_print;  //file sequence index and print frequency in # of optimization steps

//output stream
Teuchos::RCP<Teuchos::FancyOStream> fos;
int last_print_step;

//debug and system functions/variables
double CPU_Time();
void init_clock();
double initial_CPU_time, communication_time, dev2host_time, host2dev_time, output_time;


void read_mesh_ensight(char* MESH,
                       mesh_t &mesh,
                       node_t &node,
                       elem_t &elem,
                       corner_t &corner,
                       const size_t num_dims,
                       const size_t rk_num_bins);

// for string delimiter parsing
std::vector<std::string> split (std::string s, std::string delimiter);

void readVTKPn(char* MESH,
                 mesh_t &mesh,
                 node_t &node,
                 elem_t &elem,
                 zone_t &zone,
                 mat_pt_t &mat_pt,
                 corner_t &corner,
                 fe_ref_elem_t &ref_elem,
                 const size_t num_dims,
               const size_t rk_num_bins);

void input(CArrayKokkos <material_t> &material,
           CArrayKokkos <mat_fill_t> &mat_fill,
           CArrayKokkos <boundary_t> &boundary,
           CArrayKokkos <double> &state_vars,
           size_t &num_materials,
           size_t &num_fills,
           size_t &num_boundaries,
           size_t &num_dims,
           size_t &num_state_vars,
           double &dt_start,
           double &time_final,
           double &dt_max,
           double &dt_min,
           double &dt_cfl,
           double &graphics_dt_ival,
           size_t &graphics_cyc_ival,
           size_t &cycle_stop,
           size_t &rk_num_stages);

KOKKOS_FUNCTION
void get_gauss_leg_pt_jacobian(const mesh_t &mesh,
                               const elem_t &elem,
                               const fe_ref_elem_t &ref_elem,
                               const DViewCArrayKokkos <double> &node_coords,
                               CArrayKokkos <double> &gauss_legendre_jacobian,
                               CArrayKokkos <double> &gauss_legendre_det_j,
                               CArrayKokkos <double> &gauss_legendre_jacobian_inverse);

KOKKOS_FUNCTION
void get_vol(DViewCArrayKokkos <double> &elem_vol,
             const DViewCArrayKokkos <double> &node_coords,
             const CArrayKokkos <double> &legendre_weights,
             const CArrayKokkos <double> &legendre_jacobian_det,
             const mesh_t &mesh,
             const elem_t &elem,
             const fe_ref_elem_t &ref_elem);


KOKKOS_FUNCTION
void get_vol_hex(const DViewCArrayKokkos <double> &elem_vol,
                 const size_t elem_gid,
                 const DViewCArrayKokkos <double> &node_coords,
                 const ViewCArrayKokkos <size_t>  &elem_node_gids);


KOKKOS_FUNCTION
void get_vol_quad(const DViewCArrayKokkos <double> &elem_vol,
                  const size_t elem_gid,
                  const DViewCArrayKokkos <double> &node_coords,
                  const ViewCArrayKokkos <size_t>  &elem_node_gids);


KOKKOS_FUNCTION
double get_area_quad(const size_t elem_gid,
                     const DViewCArrayKokkos <double> &node_coords,
                     const ViewCArrayKokkos <size_t>  &elem_node_gids);


KOKKOS_FUNCTION
void get_bmatrix(const ViewCArrayKokkos <double> &B_matrix,
                 const size_t elem_gid,
                 const DViewCArrayKokkos <double> &node_coords,
                 const ViewCArrayKokkos <size_t>  &elem_node_gids);


KOKKOS_FUNCTION
void get_bmatrix2D(const ViewCArrayKokkos <double> &B_matrix,
                   const size_t elem_gid,
                   const DViewCArrayKokkos <double> &node_coords,
                   const ViewCArrayKokkos <size_t>  &elem_node_gids);


void setup(const CArrayKokkos <material_t> &material,
           const CArrayKokkos <mat_fill_t> &mat_fill,
           const CArrayKokkos <boundary_t> &boundary,
           mesh_t &mesh,
           elem_t &elem,
           zone_t &zone,
           mat_pt_t &mat_pt,
           fe_ref_elem_t &ref_elem,
           const DViewCArrayKokkos <double> &node_coords,
           DViewCArrayKokkos <double> &node_vel,
           DViewCArrayKokkos <double> &mat_pt_vel,
           DViewCArrayKokkos <double> &node_mass,
           const DViewCArrayKokkos <double> &elem_den,
           const DViewCArrayKokkos <double> &elem_pres,
           const DViewCArrayKokkos <double> &elem_stress,
           const DViewCArrayKokkos <double> &elem_sspd,
           const DViewCArrayKokkos <double> &elem_sie,
           const DViewCArrayKokkos <double> &mat_pt_sie,
           const DViewCArrayKokkos <double> &elem_vol,
           const DViewCArrayKokkos <double> &elem_mass,
           const DViewCArrayKokkos <size_t> &elem_mat_id,
           const DViewCArrayKokkos <double> &elem_statev,
           CArrayKokkos <double> &mat_pt_statev,
           const CArrayKokkos <double> &state_vars,
           const DViewCArrayKokkos <double> &corner_mass,
           const size_t num_fills,
           const size_t rk_num_bins,
           const size_t num_bdy_sets,
           const size_t num_materials,
           const size_t num_state_vars);


void write_outputs (const mesh_t &mesh,
                    DViewCArrayKokkos <double> &node_coords,
                    DViewCArrayKokkos <double> &node_vel,
                    DViewCArrayKokkos <double> &node_mass,
                    DViewCArrayKokkos <double> &elem_den,
                    DViewCArrayKokkos <double> &elem_pres,
                    DViewCArrayKokkos <double> &elem_stress,
                    DViewCArrayKokkos <double> &elem_sspd,
                    DViewCArrayKokkos <double> &elem_sie,
                    DViewCArrayKokkos <double> &elem_vol,
                    DViewCArrayKokkos <double> &elem_mass,
                    DViewCArrayKokkos <size_t> &elem_mat_id,
                    CArray <double> &graphics_times, 
                    size_t &graphics_id,
                    const double time_value);


void ensight(const mesh_t &mesh,
             const DViewCArrayKokkos <double> &node_coords,
             const DViewCArrayKokkos <double> &node_vel,
             const DViewCArrayKokkos <double> &node_mass,
             const DViewCArrayKokkos <double> &elem_den,
             const DViewCArrayKokkos <double> &elem_pres,
             const DViewCArrayKokkos <double> &elem_stress,
             const DViewCArrayKokkos <double> &elem_sspd,
             const DViewCArrayKokkos <double> &elem_sie,
             const DViewCArrayKokkos <double> &elem_vol,
             const DViewCArrayKokkos <double> &elem_mass,
             const DViewCArrayKokkos <size_t> &elem_mat_id,
             CArray <double> &graphics_times,
             size_t &graphics_id,
             const double time_value);

void VTKHexN(const mesh_t &mesh,
             const DViewCArrayKokkos <double> &node_coords,
             const DViewCArrayKokkos <double> &node_vel,
             const DViewCArrayKokkos <double> &node_mass,
             const DViewCArrayKokkos <double> &mat_pt_den,
             const DViewCArrayKokkos <double> &mat_pt_pres,
             const DViewCArrayKokkos <double> &mat_pt_stress,
             const DViewCArrayKokkos <double> &mat_pt_sspd,
             const DViewCArrayKokkos <double> &zone_sie,
             const DViewCArrayKokkos <double> &elem_vol,
             const DViewCArrayKokkos <double> &mat_pt_mass,
             const DViewCArrayKokkos <size_t> &elem_mat_id,
             CArray <double> &graphics_times,
             size_t &graphics_id,
             const double time_value);


void state_file(const mesh_t &mesh,
                const DViewCArrayKokkos <double> &node_coords,
                const DViewCArrayKokkos <double> &node_vel,
                const DViewCArrayKokkos <double> &mat_pt_vel,
                const DViewCArrayKokkos <double> &mat_pt_coords,
                const DViewCArrayKokkos <double> &mat_pt_h,
                const DViewCArrayKokkos <double> &node_mass,
                const DViewCArrayKokkos <double> &elem_den,
                const DViewCArrayKokkos <double> &elem_pres,
                const DViewCArrayKokkos <double> &elem_stress,
                const DViewCArrayKokkos <double> &elem_sspd,
                const DViewCArrayKokkos <double> &elem_sie,
                const DViewCArrayKokkos <double> &elem_vol,
                const DViewCArrayKokkos <double> &elem_mass,
                const DViewCArrayKokkos <size_t> &elem_mat_id,
                const double time_value );


void tag_bdys(const CArrayKokkos <boundary_t> &boundary,
              mesh_t &mesh,
              const DViewCArrayKokkos <double> &node_coords);


KOKKOS_FUNCTION
size_t check_bdy(const size_t patch_gid,
                 const int this_bc_tag,
                 const double val,
                 const mesh_t &mesh,
                 const DViewCArrayKokkos <double> &node_coords);


void build_boundry_node_sets(const CArrayKokkos <boundary_t> &boundary,
                             mesh_t &mesh);


void boundary_velocity(const mesh_t &mesh,
                       const CArrayKokkos <boundary_t> &boundary,
                       DViewCArrayKokkos <double> &node_vel,
                       const double time_value);


KOKKOS_FUNCTION
void ideal_gas(const DViewCArrayKokkos <double> &elem_pres,
               const DViewCArrayKokkos <double> &elem_stress,
               const size_t elem_gid,
               const size_t legendre_gid,
               const size_t mat_id,
               const DViewCArrayKokkos <double> &elem_state_vars,
               const DViewCArrayKokkos <double> &elem_sspd,
               const double den,
               const double sie);


KOKKOS_FUNCTION
void user_eos_model(const DViewCArrayKokkos <double> &elem_pres,
                       const DViewCArrayKokkos <double> &elem_stress,
                       const size_t elem_gid,
                       const size_t legendre_gid,
                       const size_t mat_id,
                       const DViewCArrayKokkos <double> &elem_state_vars,
                       const DViewCArrayKokkos <double> &elem_sspd,
                       const double den,
                       const double sie);


void user_model_init(const DCArrayKokkos <double> &file_state_vars,
                     const size_t num_state_vars,
                     const size_t mat_id,
                     const size_t num_elems);

void get_artificial_viscosity(CArrayKokkos <double> &sigma_a,
                              const DViewCArrayKokkos <double> &vel,
                              const DViewCArrayKokkos <double> &mat_pt_vel,
                              const DViewCArrayKokkos <double> &den,
                              const DViewCArrayKokkos <double> &sspd,
                              const DViewCArrayKokkos <double> &vol,
                              DViewCArrayKokkos <double> &mat_pt_h,
                              const CArrayKokkos <double> &legendre_jacobian_inverse,
                              const CArrayKokkos <double> &legendre_jacobian,
                              const CArrayKokkos <double> &legendre_jacobian_inverse_t0,
                              const CArrayKokkos <double> &char_length_t0,
                              const mesh_t &mesh,
                              const fe_ref_elem_t &ref_elem,
                              const size_t stage);

KOKKOS_FUNCTION
void invert_matrix(CArrayKokkos <double> &mtx_inv,
                   CArrayKokkos <double> &mtx,
                   const mesh_t &mesh,
                   const int size);

void append_artificial_viscosity(DViewCArrayKokkos <double> &sigma,
                                 const CArrayKokkos <double> &sigma_a,
                                 const mesh_t &mesh,
                                 const size_t stage);

void build_force_tensor(CArrayKokkos <double> &force_tensor,
                        const size_t stage,
                        const mesh_t &mesh,
                        const DViewCArrayKokkos <double> &stress_tensor,
                        const CArrayKokkos <double> &legendre_grad_basis,
                        const CArrayKokkos <double> &bernstein_basis,
                        const CArrayKokkos <double> &legendre_weights,
                        const CArrayKokkos <double> &legendre_jacobian_det,
                        const CArrayKokkos <double> &legendre_jacobian_inverse );

void get_stress_tensor(DViewCArrayKokkos <double> &elem_stress,
                       const size_t stage,
                       const mesh_t &mesh,
                       const DViewCArrayKokkos <double> &elem_pressure);

void get_grad_vel(CArrayKokkos <double> &grad_vel,
                              const DViewCArrayKokkos <double> &vel,
                              const CArrayKokkos <double> &legendre_jacobian_inverse,
                              const mesh_t &mesh,
                              const fe_ref_elem_t &ref_elem,
                              const size_t stage);

void get_sym_grad_vel(CArrayKokkos <double> &sym_grad_vel,
                              const DViewCArrayKokkos <double> &vel,
                              const CArrayKokkos <double> &legendre_jacobian_inverse,
                              const mesh_t &mesh,
                              const fe_ref_elem_t &ref_elem,
                              const size_t stage);

void get_sym_grad_vel(CArrayKokkos <double> &sym_grad_vel,
                              const DViewCArrayKokkos <double> &vel,
                              const CArrayKokkos <double> &legendre_jacobian_inverse,
                              const mesh_t &mesh,
                              const fe_ref_elem_t &ref_elem,
                              const size_t stage);
void get_anti_sym_grad_vel(CArrayKokkos <double> &anti_sym_grad_vel,
                              const DViewCArrayKokkos <double> &vel,
                              const CArrayKokkos <double> &legendre_jacobian_inverse,
                              const mesh_t &mesh,
                              const fe_ref_elem_t &ref_elem,
                              const size_t stage);

void get_div_vel(CArrayKokkos <double> &div_vel,
                              const DViewCArrayKokkos <double> &vel,
                              const CArrayKokkos <double> &legendre_jacobian_inverse,
                              const mesh_t &mesh,
                              const fe_ref_elem_t &ref_elem,
                              const size_t stage);


void correct_force_tensor(CArrayKokkos <double> &force_tensor,
                         const size_t stage,
                         const mesh_t &mesh,
                         const CArrayKokkos <double> &L2,
                         const CArrayKokkos <double> &M,
                         const CArrayKokkos <double> &m,
                         const CArrayKokkos <double> &F_dot_ones,
                         const double dt);

KOKKOS_FUNCTION
void assemble_kinematic_mass_matrix( CArrayKokkos <double> &M_V,
                                    CArrayKokkos <double> &lumped_mass,
                                    const mesh_t &mesh,
                                    const CArrayKokkos <double> &basis,
                                    const CArrayKokkos <double> &legendre_weights,
                                    const CArrayKokkos <double> &legendre_jacobian_det,
                                    const DViewCArrayKokkos <double> &density );

void compute_lumped_mass(CArrayKokkos <double> &lumped_mass,
                         const mesh_t &mesh,
                         const CArrayKokkos <double> &basis,
                         const CArrayKokkos <double> &legendre_weights,
                         const CArrayKokkos <double> &legendre_jacobian_det,
                         const DViewCArrayKokkos <double> &density);

void assemble_thermodynamic_mass_matrix( CArrayKokkos <double> &M,
                                    CArrayKokkos <double> &m,
                                    CArrayKokkos <double> &M_inv,
                                    const mesh_t &mesh,
                                    const CArrayKokkos <double> &basis,
                                    const CArrayKokkos <double> &legendre_weights,
                                    const CArrayKokkos <double> &legendre_jacobian_det,
                                    const DViewCArrayKokkos <double> &density );

void assemble_L2(   CArrayKokkos <double> &L2,
                    const size_t stage,
                    const double dt,
                    const mesh_t &mesh,
                    CArrayKokkos <double> &M_dot_u,
                    CArrayKokkos <double> &F_dot_ones,
                    const CArrayKokkos <double> &force_tensor,
                    const CArrayKokkos <double> &mass_matrix,
                    const DViewCArrayKokkos <double> &node_vel );

void update_momentum(DViewCArrayKokkos <double> &node_vel,
                     const size_t stage,
                     const mesh_t &mesh,
                     const double dt,
                     const CArrayKokkos <double> &L2,
                     const CArrayKokkos <double> &lumped_mass);

void update_internal_energy(DViewCArrayKokkos <double> &zone_sie,
                     const size_t stage,
                     const mesh_t &mesh,
                     const CArrayKokkos <double> &M_e_inv,
                     const CArrayKokkos <double> &force_tensor,
                     CArrayKokkos <double> &F_dot_u,
                     const CArrayKokkos <double> &Fc,
                     CArrayKokkos <double> &Fc_dot_u,
                     const CArrayKokkos <double> &source,
                     const DViewCArrayKokkos <double> &node_vel,
                     const CArrayKokkos <double> &lumped_mass,
                     const double dt);
                    //  const CArrayKokkos <double> &A1,
                    //  const CArrayKokkos <double> &lumped_mass);


void get_sie_source(CArrayKokkos <double> source,
                    const DViewCArrayKokkos <double> &node_coords,
                    const mat_pt_t &mat_pt,
                    const mesh_t &mesh,
                    const zone_t &zone,
                    const fe_ref_elem_t &ref_elem,
                    const size_t stage
                    );

void update_position_rdh(const size_t stage,
                         double dt,
                         const mesh_t &mesh,
                         DViewCArrayKokkos <double> &node_coords,
                         const DViewCArrayKokkos <double> &node_vel);

void rdh_solve(CArrayKokkos <material_t> &material,
               CArrayKokkos <boundary_t> &boundary,
               mesh_t &mesh,
               elem_t &elem,
               node_t &node,
               fe_ref_elem_t &ref_elem,
               mat_pt_t &mat_pt,
               zone_t &zone,
               DViewCArrayKokkos <double> &node_coords,
               DViewCArrayKokkos <double> &mat_pt_coords,
               DViewCArrayKokkos <double> &node_vel,
               DViewCArrayKokkos <double> &mat_pt_vel,
               CArrayKokkos <double> &M_v,
               CArrayKokkos <double> &lumped_mass,
               DViewCArrayKokkos <double> &node_mass,
               DViewCArrayKokkos <double> &mat_pt_den,
               DViewCArrayKokkos <double> &mat_pt_pres,
               DViewCArrayKokkos <double> &mat_pt_stress,
               DViewCArrayKokkos <double> &mat_pt_sspd,
               DViewCArrayKokkos <double> &zone_sie,
               DViewCArrayKokkos <double> &mat_pt_sie,
               CArrayKokkos <double> &M_e,
               CArrayKokkos <double> &zonal_mass,
               DViewCArrayKokkos <double> &elem_vol,
               DViewCArrayKokkos <double> &mat_pt_div,
               DViewCArrayKokkos <double> &mat_pt_mass,
               DViewCArrayKokkos <double> &mat_pt_h,
               DViewCArrayKokkos <size_t> &elem_mat_id,
               DViewCArrayKokkos <double> &elem_statev,
               CArrayKokkos <double> &mat_pt_statev,
               CArrayKokkos <double> &grad_vel,
               CArrayKokkos <double> &sym_grad_vel,
               CArrayKokkos <double> &anti_sym_grad_vel,
               CArrayKokkos <double> &div_vel,
               double &time_value,
               const double time_final,
               const double dt_max,
               const double dt_min,
               const double dt_cfl,
               double &graphics_time,
               size_t graphics_cyc_ival,
               double graphics_dt_ival,
               const size_t cycle_stop,
               const size_t rk_num_stages,
               double dt,
               const double fuzz,
               const double tiny,
               const double small,
               CArray <double> &graphics_times,
               size_t &graphics_id);


void get_timestep_HexN(mesh_t &mesh,
                  DViewCArrayKokkos <double> &node_coords,
                  DViewCArrayKokkos <double> &node_vel,
                  DViewCArrayKokkos <double> &mat_pt_sspd,
                  DViewCArrayKokkos <double> &elem_vol,
                  double time_value,
                  const double graphics_time,
                  const double time_final,
                  const double dt_max,
                  const double dt_min,
                  const double dt_cfl,
                  double &dt,
                  const double fuzz);

void init_tn(const mesh_t &mesh,
             DViewCArrayKokkos <double> &node_coords,
             DViewCArrayKokkos <double> &node_vel,
             DViewCArrayKokkos <double> &zone_sie,
             DViewCArrayKokkos <double> &stress);

KOKKOS_FUNCTION
double heron(const double x1,
             const double y1,
             const double x2,
             const double y2,
             const double x3,
             const double y3);


