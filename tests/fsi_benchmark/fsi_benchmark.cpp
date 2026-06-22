#include "supg_solver.h"
#include "insim.h"
#include "parameters.h"
#include "utilities.h"
#include "fsi.h"
#include "hyper_elasticity.h"
#include <cmath>

extern template class Fluid::SUPGFluidSolver<2>;
extern template class Fluid::SUPGFluidSolver<3>;
extern template class Fluid::InsIM<2>;

using namespace dealii;

template <int dim>
class BoundaryValues : public Function<dim>
{
    public:
        BoundaryValues() : Function<dim>(dim+1) {}
        virtual double value(const Point<dim> &p, const unsigned int component) const;
        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const;
};

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int component) const
{
    double left_boundary = 0.0;
    if (component == 0 && std::abs(p[0] - left_boundary) < 1e-10)
    {
        // Smooth ramp-up: 0.5*(1 - cos(pi*t/2)) for t < 2, then 1.0
        double t = this->get_time();
        double ramp_time = 2.0;
        double ramp = (t < ramp_time) ? 0.5 * (1.0 - std::cos(M_PI * t / ramp_time)) : 1.0;

        // Parabolic inlet profile
        double U_avg = 2.0; // Mean velocity 
        double U = 1.5 * U_avg * 4 * p[1] * (0.41 - p[1]) / 0.1681;
        if (dim == 3)
        {
            U *= 1.5 * 4 * p[2] * (0.41 - p[2]) / 0.1681;
        }
        return ramp * U;
    }
    return 0;
}

template <int dim>
void BoundaryValues<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
{
    for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = BoundaryValues::value(p, c);
}

int main(int argc, char *argv[])
{
    try 
    {
        std::string infile("parameters.prm");
        if (argc > 1)
        {
            infile = argv[1];
        }
        Parameters::AllParameters params(infile);

        if (params.dimension == 2)
        {
            Triangulation<2> fluid_tria;
            Utils::GridCreator<2>::flow_around_cylinder(fluid_tria); // uses mesh from flow around the cylinder case for the rigid cylinder part

            // // === Local refinement near the cylinder + flap region ===
            // {
            //     const unsigned int n_local_passes = 1;

            //     // Box 1: square around the cylinder (centered at 0.2, 0.2)
            //     // Half-width = 0.10 → spans x ∈ [0.10, 0.30], y ∈ [0.10, 0.30]
            //     const double cyl_xmin = 0.10, cyl_xmax = 0.30;
            //     const double cyl_ymin = 0.10, cyl_ymax = 0.30;

            //     // Box 2: rectangle around the flap (centered at flap centerline y = 0.20)
            //     // Spans x ∈ [0.25, 0.65] (flap + a little wake), y ∈ [0.16, 0.24] (flap + 2cm above/below)
            //     const double flap_xmin = 0.25, flap_xmax = 0.70;
            //     const double flap_ymin = 0.12, flap_ymax = 0.28;

            //     for (unsigned int pass = 0; pass < n_local_passes; ++pass)
            //     {
            //         for (auto &cell : fluid_tria.active_cell_iterators())
            //         {
            //             // Check the cell CENTER, not its vertices.
            //             // This prevents the "spillover" where a cell with one corner
            //             // in the region gets refined and visually extends the box.
            //             const Point<2> c = cell->center();

            //             const bool in_cyl_box =
            //                 (c[0] >= cyl_xmin && c[0] <= cyl_xmax &&
            //                 c[1] >= cyl_ymin && c[1] <= cyl_ymax);

            //             const bool in_flap_box =
            //                 (c[0] >= flap_xmin && c[0] <= flap_xmax &&
            //                 c[1] >= flap_ymin && c[1] <= flap_ymax);

            //             if (in_cyl_box || in_flap_box)
            //                 cell->set_refine_flag();
            //         }
            //         fluid_tria.execute_coarsening_and_refinement();
            //     }

            //     std::cout << "Fluid mesh after local refinement: "
            //             << fluid_tria.n_active_cells() << " active cells" << std::endl;
            // }

            auto ptr = std::make_shared<BoundaryValues<2>>(BoundaryValues<2>());
            Fluid::SUPGFluidSolver<2> fluid(fluid_tria, params, ptr);
            
            Triangulation<2> solid_tria;
            // setting up the flag geometry
            double flag_x0 = 0.2489;
            double flag_x1 = 0.6;
            double flag_y0 = 0.19;
            double flag_y1 = 0.21; 
            double h = 0.008; // mesh size  was 0.0025 initial run
            double l = 0.010; 
            dealii::GridGenerator::subdivided_hyper_rectangle(
                solid_tria,
                {static_cast<unsigned int>((flag_x1 - flag_x0) / l),
                 static_cast<unsigned int>((flag_y1 - flag_y0) / h)},
                Point<2>(flag_x0, flag_y0),
                Point<2>(flag_x1, flag_y1),
                true);
            Solid::HyperElasticity<2> solid(solid_tria, params);

            // to output mesh ratio between fluid and solid for debugging
            std::cout << "Fluid mesh: " << fluid_tria.n_active_cells() << " active cells" << std::endl;
            std::cout << "Solid mesh: " << solid_tria.n_active_cells() << " active cells" << std::endl;
        
            FSI<2> fsi(fluid, solid, params, false);
            fsi.run();
        }
      else
        {
            AssertThrow(false, ExcNotImplemented());     
        }
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}