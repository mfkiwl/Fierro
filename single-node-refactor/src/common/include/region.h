/**********************************************************************************************
� 2020. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and
to permit others to do so.
This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
1.  Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.
3.  Neither the name of the copyright holder nor the names of its contributors may be used
to endorse or promote products derived from this software without specific prior
written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************************/

#ifndef FIERRO_REGION_H
#define FIERRO_REGION_H

#include <map>

#include "matar.h"

#include "state.h"

#include "initial_conditions.h"

// ==============================================================================
//   Fierro material regions
// ==============================================================================
namespace region
{
    // for tagging volumes to paint material onto the mesh
    enum vol_tag
    {
        no_volume = 0,
        global = 1,             ///< tag every elements in the mesh
        box = 2,                ///< tag all elements inside a box
        cylinder = 3,           ///< tag all elements inside a cylinder
        sphere = 4,             ///< tag all elements inside a sphere
        readVoxelFile = 5,      ///< tag all elements in a voxel mesh (structured VTK)
        readPolycrystalFile = 6,///< tag all elements in a polycrystallince voxel mesh (structured VTK)
        readSTLFile = 7,        ///< read a STL file and voxelize it
        readVTUFile = 8,        ///< tag elements in an unstructured .vtu mesh with object_ids
    };
    
} // end of namespace

static std::map<std::string, region::vol_tag> region_type_map
{
    { "global", region::global },
    { "box", region::box },
    { "sphere", region::sphere },
    { "cylinder", region::cylinder },
    { "voxel_file", region::readVoxelFile },
    { "vtu_file", region::readVTUFile}
};


/////////////////////////////////////////////////////////////////////////////
///
/// \struct RegionFill_t
///
/// \brief Geometry data for regions of materials/states
///
/////////////////////////////////////////////////////////////////////////////
struct RegionFill_t
{
    // type
    region::vol_tag volume; ///< Type of volume for this region eg. global, box, sphere, planes, etc.

    // solver id
    size_t solver_id; ///< solver ID for this region

    // material id
    size_t material_id; ///< Material ID for this region

    // planes
    double x1 = 0.0; ///< First X plane for creating a box
    double x2 = 0.0; ///< Second X plane for creating a box
    double y1 = 0.0; ///< First Y plane for creating a box
    double y2 = 0.0; ///< Second Y plane for creating a box
    double z1 = 0.0; ///< First Z plane for creating a box
    double z2 = 0.0; ///< Second Z plane for creating a box

    // radius
    double radius1 = 0.0;   ///< Inner radius to fill for sphere
    double radius2 = 0.0;   ///< Outer radius to fill for sphere

    // initial condition for velocity 
    init_conds::init_vector_conds vel_field = init_conds::noICsVec;  ///< ICs for velocity in this region

    // initial conditions for density
    init_conds::init_scalar_conds den_field = init_conds::noICsScalar;

    // initial conditions for specific internal energy
    init_conds::init_scalar_conds sie_field = init_conds::noICsScalar;

    // initial conditions for specific internal energy
    init_conds::init_scalar_conds ie_field = init_conds::noICsScalar;

    // initial conditions for level set field
    init_conds::init_scalar_conds level_set_field = init_conds::noICsScalar;

    // initial condition for temperature distribution
    init_conds::init_scalar_conds temperature_field= init_conds::noICsScalar;

    // initial condition for thermal conductivity distribution
    init_conds::init_scalar_conds thermal_conductivity_field= init_conds::noICsScalar;

    // initial condition for specific heat distribution
    init_conds::init_scalar_conds specific_heat_field= init_conds::noICsScalar;

    // initial condition for volume fraction distribution
    init_conds::init_scalar_conds volfrac_field = init_conds::uniform;

    // velocity coefficients by component
    double u = 0.0; ///< U component of velocity
    double v = 0.0; ///< V component of velocity
    double w = 0.0; ///< W component of velocity

    double speed = 0.0; ///< velocity magnitude for radial velocity initialization

    double temperature = 0.0; ///< temperature magnitude for initialization
    double temperature_origin[3] = { 0.0, 0.0, 0.0 }; ///< Origin for temperature field

    double ie  = 0.0;  ///< extensive internal energy
    double sie = 0.0;  ///< specific internal energy
    double sie_origin[3] = { 0.0, 0.0, 0.0 }; ///< Origin for sie or ie field

    double den = 0.0;  ///< density
    double den_origin[3] = { 0.0, 0.0, 0.0 }; ///< Origin for den field


    double level_set = 0.0; ///< level set field
    double level_set_slope = 0.0; ///< slope of level_set field
    double level_set_origin[3] = { 0.0, 0.0, 0.0 }; ///< Origin for level_set field
    
    // note: setup applies min and max fcns, making it [0:1]
    double volfrac = 1.0; ///< volume fraction of material field
    double volfrac_slope = 0.0; ///< slope of volume fraction field
    double volfrac_origin[3] = { 0.0, 0.0, 0.0 }; ///< Origin for volume fraction field

    double specific_heat = 0.0; ///< specific heat
    double specific_heat_origin[3] = { 0.0, 0.0, 0.0 }; ///< Origin for specific heat field

    double thermal_conductivity = 0.0; ///< thermal conductivity
    double thermal_conductivity_origin[3] = { 0.0, 0.0, 0.0 }; ///< Origin for thermal cond field


    // the volume origin
    double origin[3] = { 0.0, 0.0, 0.0 }; ///< Origin for region fill, its the volume origin

    int part_id = 1; // object_id in the .vtu file, starts at 1 and goes to N parts
};

/////////////////////////////////////////////////////////////////////////////
///
/// \struct RegionFill_host_t
///
/// \brief Geometry data, on the cpu only, for regions of materials/states
///
/////////////////////////////////////////////////////////////////////////////
struct RegionFill_host_t
{
    std::string file_path; ///< path of mesh file

    // scale parameters for input mesh files
    double scale_x = 1.0;
    double scale_y = 1.0;
    double scale_z = 1.0;
};


/////////////////////////////////////////////////////////////////////////////
///
/// \struct SolverRegionSetup_t
///
/// \brief Contains kokkos arrays of fill instructions for the regions
///
/////////////////////////////////////////////////////////////////////////////
struct SolverRegionSetup_t
{
    mtr::DCArrayKokkos<size_t> reg_fills_in_solver;     // (solver_id, fill_lid)
    mtr::DCArrayKokkos<size_t> num_reg_fills_in_solver; // (solver_id)

    mtr::CArrayKokkos<RegionFill_t> region_fills;      ///< Region data for simulation mesh, set the initial conditions
    mtr::CArray<RegionFill_host_t> region_fills_host;  ///< Region data on CPU, set the initial conditions

    // vectors storing what state is to be filled on the mesh
    std::vector <fill_gauss_state> fill_gauss_states; ///< Enums for the state at guass_pts, which live under the mat_pts
    std::vector <fill_node_state>  fill_node_states;  ///< Enums for the state at nodes
};


// ----------------------------------
// valid inputs for a material fill
// ----------------------------------
static std::vector<std::string> str_region_inps
{
    "volume",
    "solver_id",
    "material_id",
    "volume_fraction",
    "velocity",
    "temperature",
    "density",
    "specific_heat",
    "thermal_conductivity",
    "specific_internal_energy",
    "internal_energy",
    "level_set"
};

// ---------------------------------------------------------
// valid inputs for volume, these are subfields under volume
// ---------------------------------------------------------
static std::vector<std::string> str_region_volume_inps
{
    "type",
    "file_path",
    "x1",
    "x2",
    "y1",
    "y2",
    "z1",
    "z2",
    "radius1",
    "radius2",
    "scale_x",
    "scale_y",
    "scale_z",
    "origin",
    "part_id"
};

// ---------------------------------------------------------------------
// valid inputs for filling velocity, these are subfields under velocity
// ---------------------------------------------------------------------
static std::vector<std::string> str_region_vel_inps
{
    "type",
    "u",
    "v",
    "w",
    "speed"
};

// ---------------------------------------------------------------------
// valid inputs for filling den, these are subfields under den
// ---------------------------------------------------------------------
static std::vector<std::string> str_region_den_inps
{
    "type",
    "value"
};

// ---------------------------------------------------------------------
// valid inputs for filling sie, these are subfields under sie
// ---------------------------------------------------------------------
static std::vector<std::string> str_region_sie_inps
{
    "type",
    "value"
};

// ---------------------------------------------------------------------
// valid inputs for filling ie, these are subfields under ie
// ---------------------------------------------------------------------
static std::vector<std::string> str_region_ie_inps
{
    "type",
    "value"
};

// ---------------------------------------------------------------------
// valid inputs for filling temperature, these are subfields under tempature
// ---------------------------------------------------------------------
static std::vector<std::string> str_region_temperature_inps
{
    "type",
    "value"
};

// ---------------------------------------------------------------------
// valid inputs for filling specific heat, these are subfields under specific heat
// ---------------------------------------------------------------------
static std::vector<std::string> str_region_specific_heat_inps
{
    "type",
    "value"
};

// ---------------------------------------------------------------------
// valid inputs for filling thermal conductivity, these are subfields under thermal conductivity
// ---------------------------------------------------------------------
static std::vector<std::string> str_region_thermal_conductivity_inps
{
    "type",
    "value"
};

// ---------------------------------------------------------------------
// valid inputs for filling level set, these are subfields under level_set
// ---------------------------------------------------------------------
static std::vector<std::string> str_region_level_set_inps
{
    "type",
    "value",
    "slope",
    "origin"
};


// ---------------------------------------------------------------------
// valid inputs for filling volfrac, these are subfields under volume fuction
// ---------------------------------------------------------------------
static std::vector<std::string> str_region_volfrac_inps
{
    "type",
    "value",
    "slope",
    "origin"
};

// ----------------------------------
// required inputs for region options
// ----------------------------------
static std::vector<std::string> region_required_inps
{
    "material_id",
    "volume"
};

// -------------------------------------
// required inputs for specifying volume
// -------------------------------------
static std::vector<std::string> region_volume_required_inps
{
    "type"
};


// -------------------------------------
// required inputs for filling velocity
// -------------------------------------
static std::vector<std::string> region_vel_required_inps
{
    "type"
};

// -------------------------------------
// required inputs for filling density
// -------------------------------------
static std::vector<std::string> region_den_required_inps
{
    "type"
};

// -------------------------------------
// required inputs for filling sie
// -------------------------------------
static std::vector<std::string> region_sie_required_inps
{
    "type"
};

// -------------------------------------
// required inputs for filling ie
// -------------------------------------
static std::vector<std::string> region_ie_required_inps
{
    "type"
};

// -------------------------------------
// required inputs for filling temperature
// -------------------------------------
static std::vector<std::string> region_temperature_required_inps
{
    "type"
};

// -------------------------------------
// required inputs for filling specific heat
// -------------------------------------
static std::vector<std::string> region_specific_heat_required_inps
{
    "type"
};

// -------------------------------------
// required inputs for filling thermal conductivity
// -------------------------------------
static std::vector<std::string> region_thermal_conductivity_required_inps
{
    "type"
};

// -------------------------------------
// required inputs for filling level set
// -------------------------------------
static std::vector<std::string> region_level_set_required_inps
{
    "type"
//    "value",
//    "slope"
};

// -------------------------------------
// required inputs for filling volume fraction
// -------------------------------------
static std::vector<std::string> region_volfrac_required_inps
{
    "type"
};

#endif // end Header Guard