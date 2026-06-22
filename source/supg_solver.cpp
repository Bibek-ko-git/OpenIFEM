#include "supg_solver.h"
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/numerics/vector_tools.h>


namespace Fluid
{
  using namespace dealii;


  // ========================================================================
  // BlockTriangularPreconditioner Implementation to replace ILUPreconditioner
  // ========================================================================


  template <int dim>
  void SUPGFluidSolver<dim>::BlockTriangularPreconditioner::initialize(
      const BlockSparseMatrix<double> &system_matrix)
  {
    // Use stronger ILU for better conditioning
    typename SparseILU<double>::AdditionalData ilu_data;
    ilu_data.strengthen_diagonal = 0.1;  // Add diagonal strengthening
    // the ILU velocity abd pressure blocks
    ilu_velocity.initialize(system_matrix.block(0, 0), ilu_data);
    ilu_pressure.initialize(system_matrix.block(1, 1), ilu_data);


    // coupling blocks added
    Avp = &system_matrix.block(0,1);
    Apv = &system_matrix.block(1,0);
  }


  // ========================================================================
  // Block Triangular Preconditioner with Coupling
  // ========================================================================
  template <int dim>
  void SUPGFluidSolver<dim>::BlockTriangularPreconditioner::vmult(
      BlockVector<double> &dst, const BlockVector<double> &src) const
  {
    // // Block Triangular preconditioner
    // // P^{-1} = [A^{-1}    -A^{-1}*B^T*S^{-1}]
    // //          [0          S^{-1}            ]
    // // With SUPG: S = App (contains PSPG Laplacian)
    Vector<double> tmp(src.block(0).size());

    // step 1. Approximate pressure solve
    ilu_pressure.vmult(dst.block(1), src.block(1));

    // step 2. velocity solve with pressure coupling correction
    Avp -> vmult(tmp, dst.block(1));
    tmp *= -1.0;
    tmp += src.block(0);
    ilu_velocity.vmult(dst.block(0), tmp);
  }
  // ========================================================================
  // SUPGFluidSolver Implementation
  // ========================================================================


    template <int dim>
    SUPGFluidSolver<dim>::SUPGFluidSolver(Triangulation<dim> &tria,
                      const Parameters::AllParameters &parameters,
                      std::shared_ptr<Function<dim>> bc)
      : FluidSolver<dim>(tria, parameters, bc)
    {
      // No need to assert any condition here, as PSPG can handle equal order elements.
    }


  template <int dim>
  void SUPGFluidSolver<dim>::initialize_system()
  {
    FluidSolver<dim>::initialize_system();
    newton_update.reinit(dofs_per_block);
    evaluation_point.reinit(dofs_per_block);
    preconditioner = std::make_shared<BlockTriangularPreconditioner>();

    // Setup cell properties for FSI
    setup_cell_property();
   
    // Initialize stress storage
    stress.resize(dim);
    for (unsigned int i = 0; i < dim; ++i)
    {
      stress[i].resize(dim);
      for (unsigned int j = 0; j < dim; ++j)
      {
        stress[i][j].reinit(scalar_dof_handler.n_dofs());
      }
    }
  }

  template <int dim>
  void SUPGFluidSolver<dim>::assemble(const bool use_nonzero_constraints)
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");

    system_matrix = 0;
    system_rhs = 0;

    FEValues<dim> fe_values(fe, volume_quad_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
                       
    FEFaceValues<dim> fe_face_values(fe, face_quad_formula,
                                     update_values | update_normal_vectors |
                                     update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = volume_quad_formula.size();
    const unsigned int n_face_q_points = face_quad_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    // Current iteration (k)
    std::vector<Tensor<1, dim>> u_k(n_q_points);
    std::vector<Tensor<2, dim>> grad_u_k(n_q_points);
    std::vector<double>         p_k(n_q_points);
    std::vector<Tensor<1, dim>> grad_p_k(n_q_points);

    // Previous time step (n)
    std::vector<Tensor<1, dim>> u_n(n_q_points);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);
    std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

    const double rho = parameters.fluid_rho;
    const double mu  = parameters.viscosity;
    const double nu  = mu/rho;
    const double dt  = time.get_delta_t();

    Tensor<1, dim> gravity;
    for (unsigned int d = 0; d < dim; ++d)
      gravity[d] = parameters.gravity[d];

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // Get cell indicator for FSI
      auto p = cell_property.get_data(cell);
      const int ind = p[0]->indicator;
     
      // // Density varies with indicator (fluid vs solid region)
      // const double rho = parameters.fluid_rho +
      //                   ind * (parameters.solid_rho - parameters.fluid_rho);
      // const double nu = mu / rho;

      fe_values.reinit(cell);
      local_matrix = 0;
      local_rhs = 0;

      // Get solution values at quadrature points
      fe_values[velocities].get_function_values(evaluation_point, u_k);
      fe_values[velocities].get_function_gradients(evaluation_point, grad_u_k);
      fe_values[pressure].get_function_values(evaluation_point, p_k);
      fe_values[pressure].get_function_gradients(evaluation_point, grad_p_k);
      fe_values[velocities].get_function_values(present_solution, u_n);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double JxW = fe_values.JxW(q);

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_u[k]      = fe_values[velocities].value(k, q);
          grad_phi_u[k] = fe_values[velocities].gradient(k, q);
          div_phi_u[k]  = fe_values[velocities].divergence(k, q);
          phi_p[k]      = fe_values[pressure].value(k, q);
          grad_phi_p[k] = fe_values[pressure].gradient(k, q);
        }

        // ================================================================
        // Stabilization parameters (Tezduyar)
        // ================================================================
        double h = cell->diameter();
        double u_mag = std::max(u_k[q].norm(), 1e-12);
       
        double tau_SUPG = 1.0 / std::sqrt(std::pow(2.0/dt, 2) +
                                          std::pow(2.0*u_mag/h, 2) +
                                          std::pow(4.0*nu/(h*h), 2));
        double tau_PSPG = tau_SUPG/rho;
       
        // LSIC with local Reynolds number limiting
        double Re_local = u_mag * h / (2.0 * nu + 1e-12);
        double z = std::min(Re_local / 3.0, 1.0);
        double tau_LSIC = (h / 2.0) * u_mag * z;
        // double tau_LSIC = std::max((h / 2.0) * u_mag * z, 1e-4 * h * h);
       
        // ================================================================
        // Strong Residuals
        // ================================================================
        // double epsilon_p = 1e-6 / dt;
        // Momentum: R_mom
        // Note: Viscous term  is OMITTED (standard in SUPG)
        Tensor<1, dim> R_mom = rho * (u_k[q] - u_n[q]) / dt;  // Time derivative
        R_mom += rho * (grad_u_k[q] * u_k[q]);                // Convection
        R_mom += grad_p_k[q];                                  // Pressure gradient
        R_mom -= rho * gravity;                                // Body force

        // Add FSI body force to strong residual for consistency
        if (ind == 1)
        {
          R_mom -= p[0]->fsi_acceleration; 
        }
       
        // Continuity: R_cont
        double R_cont = trace(grad_u_k[q]);


        // ================================================================
        // Assembly
        // ================================================================
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // ============================================================
          // RHS: Negative residual
          // ============================================================
          // Standard Galerkin terms
          double rhs_gal = rho * ((u_k[q] - u_n[q]) * phi_u[i]) / dt;
          rhs_gal += rho * (grad_u_k[q] * u_k[q]) * phi_u[i];
          rhs_gal += mu * scalar_product(grad_u_k[q], grad_phi_u[i]);
          rhs_gal -= p_k[q] * div_phi_u[i];
          rhs_gal -= rho * gravity * phi_u[i];
         
          // Continuity
          double rhs_cont = R_cont * phi_p[i];
         
          // SUPG stabilization:
          Tensor<1, dim> supg_test = u_k[q] * grad_phi_u[i];
          double rhs_supg = tau_SUPG * (supg_test * R_mom);
         
          // PSPG stabilization:
          double rhs_pspg = tau_PSPG * (grad_phi_p[i] * R_mom);
         
          // LSIC stabilization:
          double rhs_lsic = tau_LSIC * div_phi_u[i] * R_cont;

          local_rhs(i) -= (rhs_gal + rhs_cont + rhs_supg + rhs_pspg +
                          rhs_lsic) * JxW;

          // Pressure mass term in residual
          // double rhs_p_mass = epsilon_p * p_k[q] * phi_p[i];=

          // local_rhs(i) -= (rhs_gal + rhs_cont + rhs_supg + rhs_pspg +
          //                 rhs_lsic + rhs_p_mass) * JxW;


          // Critical: get stess and acceleration from the solid
          if (ind == 1)
          {
            local_rhs(i) +=
              (scalar_product(grad_phi_u[i], p[0]->fsi_stress) +
              p[0]->fsi_acceleration * phi_u[i]) * JxW;
          }
          // ============================================================
          // Matrix: Consistent Newton linearization (Jacobian)
          // ============================================================
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            // Standard Galerkin terms
            double mat_time = rho * phi_u[j] * phi_u[i] / dt;
           
            double mat_conv = rho * ((grad_phi_u[j] * u_k[q]) * phi_u[i] +
                                    (grad_u_k[q] * phi_u[j]) * phi_u[i]);
           
            double mat_visc = mu * scalar_product(grad_phi_u[j], grad_phi_u[i]);


            double mat_pres = -phi_p[j] * div_phi_u[i];
            double mat_cont = div_phi_u[j] * phi_p[i];


            // Stabilization: derivative of residual
            Tensor<1, dim> dR_mom_du;
            dR_mom_du = rho * phi_u[j] / dt;
            dR_mom_du += rho * (grad_phi_u[j] * u_k[q]);
            dR_mom_du += rho * (grad_u_k[q] * phi_u[j]);
            // dR_mom_du += grad_phi_p[j];
           
            Tensor<1, dim> dR_mom_dp;
            dR_mom_dp = grad_phi_p[j];

            double dR_cont = div_phi_u[j];

            // SUPG:
            double mat_supg = tau_SUPG * (supg_test * dR_mom_du);
            mat_supg += tau_SUPG * ( supg_test * dR_mom_dp);
            mat_supg += tau_SUPG * ((phi_u[j] * grad_phi_u[i]) * R_mom); // Additional term
           
            // PSPG:
            double mat_pspg = tau_PSPG * (grad_phi_p[i] * dR_mom_du);
            mat_pspg += tau_PSPG * (grad_phi_p[i] * dR_mom_dp);
           
            // LSIC:
            double mat_lsic = tau_LSIC * div_phi_u[i] * dR_cont;

            local_matrix(i, j) += (mat_time + mat_conv + mat_visc + mat_pres +
                                  mat_cont + mat_supg + mat_pspg + mat_lsic )* JxW;
            // double mat_p_mass = epsilon_p * phi_p[i] * phi_p[j];


            // local_matrix(i, j) += (mat_time + mat_conv + mat_visc + mat_pres +
            //                       mat_cont + mat_supg + mat_pspg + mat_lsic + mat_p_mass) * JxW;
          }
        }
      }


    // Neumann BCs
    if (parameters.n_fluid_neumann_bcs != 0)
    {
      for (unsigned int face_n = 0;
            face_n < GeometryInfo<dim>::faces_per_cell;
            ++face_n)
        {
          if (cell->at_boundary(face_n) &&
              parameters.fluid_neumann_bcs.find(
                cell->face(face_n)->boundary_id()) !=
                parameters.fluid_neumann_bcs.end())
            {
              fe_face_values.reinit(cell, face_n);
              unsigned int p_bc_id =
                cell->face(face_n)->boundary_id();
              double boundary_values_p =
                parameters.fluid_neumann_bcs[p_bc_id];
              for (unsigned int q = 0; q < n_face_q_points; ++q)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      local_rhs(i) +=
                        -(fe_face_values[velocities].value(i, q) *
                          fe_face_values.normal_vector(q) *
                          boundary_values_p * fe_face_values.JxW(q));
                    }
                }
            }
        }
    }


    // need to check this inhomogeneous constraint application
    cell->get_dof_indices(local_dof_indices);
    const AffineConstraints<double> &constraints_used =
        use_nonzero_constraints ? nonzero_constraints : zero_constraints;
    constraints_used.distribute_local_to_global(local_matrix, local_rhs,
                                                  local_dof_indices,
                                                  system_matrix, system_rhs, true);
    }
  }



  template <int dim>
  std::pair<unsigned int, double>
  SUPGFluidSolver<dim>::solve(const bool use_nonzero_constraints) // Not used here
  {
    TimerOutput::Scope timer_section(timer, "Solve linear system");


    preconditioner->initialize(system_matrix);


    // SolverControl solver_control(system_matrix.m(), 1e-6* system_rhs.l2_norm());
    SolverControl solver_control(
    std::min(static_cast<unsigned int>(5000), system_matrix.m()),  // Max 5000 iterations
    std::max(1e-6 * system_rhs.l2_norm(), 1e-10),  // Can be relaxed back to 1e-6
    true);


    GrowingVectorMemory<BlockVector<double>> vector_memory;
    SolverFGMRES<BlockVector<double>> fgmres(solver_control, vector_memory,
                                              SolverFGMRES<BlockVector<double>>::AdditionalData(100));


    newton_update = 0;


    try
    {
      fgmres.solve(system_matrix, newton_update, system_rhs, *preconditioner);
    }
    catch (const SolverControl::NoConvergence &)
    {
      std::cout << "  Warning: Linear solver did not fully converge" << std::endl;
    }


    // Important : sets the constraints on the solution update
    // first takes the inlet as non-zero input
    // then is zero since FSI handles the increment on input boundaries
    const AffineConstraints<double> &constraints_used =
      use_nonzero_constraints ? nonzero_constraints : zero_constraints;
    constraints_used.distribute(newton_update);


    return {solver_control.last_step(), solver_control.last_value()};
  }

  template <int dim>
  void SUPGFluidSolver<dim>::run_one_step(bool apply_nonzero_constraints,
                                          bool assemble_system)
  {
      (void)assemble_system;

      if (time.get_timestep() == 0)
      {
          output_results(0);
      }
      std::cout.precision(6);
      time.increment();

      std::cout << std::string(96, '*') << std::endl
                << "Time step = " << time.get_timestep() 
                << ", at t = " << std::scientific << time.current() << std::endl;
      
      evaluation_point = present_solution;
      double current_residual = 1.0;
      double initial_residual = 1.0;
      double relative_residual = 1.0;
      unsigned int outer_iteration = 0;

      while (relative_residual > parameters.fluid_tolerance && 
             current_residual > 1e-10 &&
             outer_iteration < parameters.fluid_max_iterations)
      {
          newton_update = 0.0;
          
          assemble(apply_nonzero_constraints && outer_iteration == 0);

          double res_mom  = system_rhs.block(0).l2_norm();
          double res_cont = system_rhs.block(1).l2_norm();
          std::cout << "    res_mom=" << res_mom 
                    << "  res_cont=" << res_cont 
                    << "  ratio=" << res_cont / (res_mom + 1e-30) << std::endl;
          
          current_residual = system_rhs.l2_norm();
          
          if (outer_iteration == 0)
              initial_residual = current_residual;
          
          relative_residual = current_residual / initial_residual;
          
          if (current_residual < 1e-10)
          {
              std::cout << "  Converged (residual < 1e-10)" << std::endl;
              break;
          }

          auto state = solve(apply_nonzero_constraints && outer_iteration == 0);
          
          // // Debug output
          // std::cout << "  DEBUG: ||rhs|| = " << system_rhs.l2_norm() 
          //           << ", ||du|| = " << newton_update.block(0).l2_norm()
          //           << ", ||dp|| = " << newton_update.block(1).l2_norm() << std::endl;

          evaluation_point.add(1.0, newton_update);

          std::cout << std::scientific << std::left 
                    << "  ITR = " << std::setw(2) << outer_iteration
                    << " ABS_RES = " << current_residual
                    << " REL_RES = " << relative_residual
                    << " GMRES = " << state.first << std::endl;
          
          outer_iteration++;
      }
      
      // Update solution increment (for FSI coupling)
      solution_increment = evaluation_point;
      solution_increment -= present_solution;
      
      present_solution = evaluation_point;
      
      std::cout << "  Solution L2 norm: " << present_solution.l2_norm() 
                << ", Increment norm: " << solution_increment.l2_norm() << std::endl;
      
      double max_vel = 0;
      for (unsigned int i = 0; i < dofs_per_block[0]; ++i)
          max_vel = std::max(max_vel, std::abs(present_solution.block(0)[i]));
      std::cout << "  Max velocity: " << max_vel << std::endl;

      // also the mean x-velocity for debugging
      double mean_velocity = 0.0;
      for (unsigned int i = 0; i < dofs_per_block[0]; ++i)
          mean_velocity += present_solution.block(0)[i];
      mean_velocity /= dofs_per_block[0];
      std::cout << "  Mean velocity: " << mean_velocity * 2.0 << std::endl;

      update_stress();
      
      if (time.time_to_output())
          output_results(time.get_timestep());
      if (parameters.simulation_type == "Fluid" && time.time_to_refine())
          refine_mesh(1, 3);
  }



  template <int dim>
  void SUPGFluidSolver<dim>::run()
  {
    // triangulation.refine_global(parameters.global_refinements[0]);
    setup_dofs();
    make_constraints();
    initialize_system();


    output_results(0);


    run_one_step(true);

    while (time.end() - time.current() > 1e-12)
      {
        if (parameters.use_hard_coded_values)
          {
            // Only for time dependent BCs!
            // Advance the time by delta_t and make constraints
            boundary_values->set_time(time.current() + time.get_delta_t());
            make_constraints();
            run_one_step(true);
          }
        else
          run_one_step(false);
      }
  }


  template class SUPGFluidSolver<2>;
  template class SUPGFluidSolver<3>;


} // namespace Fluid