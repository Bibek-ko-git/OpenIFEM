#ifndef SUPG_SOLVER_H
#define SUPG_SOLVER_H

#include "fluid_solver.h"

template <int>
class FSI;

namespace Fluid
{
  using namespace dealii;

  extern template class FluidSolver<2>;
  extern template class FluidSolver<3>;

  /**
   * SUPG-stabilized incompressible Navier-Stokes solver.
   * 
   * This solver implements the standard incompressible Navier-Stokes equations
   * with SUPG (Streamline Upwind Petrov-Galerkin), PSPG (Pressure-Stabilizing
   * Petrov-Galerkin), and LSIC (Least-Squares on Incompressibility Constraint)
   * stabilization for equal-order velocity-pressure interpolation.
   * 
   * Solved using Newton-Raphson iteration with a block triangular
   * preconditioner for the linearized system.
   */
  template <int dim>
  class SUPGFluidSolver : public FluidSolver<dim>
  {
  public:
    friend FSI<dim>;

    SUPGFluidSolver(Triangulation<dim> &,
                    const Parameters::AllParameters &,
                    std::shared_ptr<Function<dim>> bc =
                    std::make_shared<Functions::ZeroFunction<dim>>(
                    Functions::ZeroFunction<dim>(dim + 1)));
    virtual ~SUPGFluidSolver() {}
    void run();

  protected:
    using FluidSolver<dim>::timer;
    using FluidSolver<dim>::triangulation;
    using FluidSolver<dim>::fe;
    using FluidSolver<dim>::dof_handler;
    using FluidSolver<dim>::volume_quad_formula;
    using FluidSolver<dim>::face_quad_formula;
    using FluidSolver<dim>::zero_constraints;
    using FluidSolver<dim>::nonzero_constraints;
    using FluidSolver<dim>::sparsity_pattern;
    using FluidSolver<dim>::system_matrix;
    using FluidSolver<dim>::system_rhs;
    using FluidSolver<dim>::present_solution;
    using FluidSolver<dim>::solution_increment;
    using FluidSolver<dim>::dofs_per_block;
    using FluidSolver<dim>::time;
    using FluidSolver<dim>::parameters;
    using FluidSolver<dim>::setup_dofs;
    using FluidSolver<dim>::make_constraints;
    using FluidSolver<dim>::output_results;
    using FluidSolver<dim>::refine_mesh;
    using FluidSolver<dim>::boundary_values;
    using FluidSolver<dim>::update_stress;
    using FluidSolver<dim>::setup_cell_property;
    using FluidSolver<dim>::cell_property;
    using FluidSolver<dim>::scalar_dof_handler;
    using FluidSolver<dim>::stress;

    // We do NOT declare these again. We access them via this-> or simple name if visible.
    // If they are private in base, we can't access them. Assuming they are protected.
    void initialize_system() override;
    void assemble(const bool use_nonzero_constraints);
    std::pair<unsigned int, double> solve(const bool use_nonzero_constraints);
    void run_one_step(bool apply_nonzero_constraints, 
                      bool assemble_system = true) override;

    class BlockTriangularPreconditioner : public Subscriptor
    {
    public:
      void initialize(const BlockSparseMatrix<double> &system_matrix);
      void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;

    private:
      SparseILU<double> ilu_velocity;
      SparseILU<double> ilu_pressure;
      SmartPointer<const SparseMatrix<double>> Avp;  // Block(0,1)
      SmartPointer<const SparseMatrix<double>> Apv;  // Block(1,0)
    };

    std::shared_ptr<BlockTriangularPreconditioner> preconditioner;

    BlockVector<double> newton_update;
    BlockVector<double> evaluation_point;

  };
} // namespace Fluid

#endif // SUPG_SOLVER_H
