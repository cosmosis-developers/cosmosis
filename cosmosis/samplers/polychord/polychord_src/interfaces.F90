!> This allows for a simple C interface... 


module interfaces_module
    use iso_c_binding
    use utils_module, only: dp
    use priors_module, only: prior, hypercube_to_physical

    implicit none
    private
    public :: run_polychord, polychord_c_interface, polychord_c_interface_ini

    ! Nothing in this module may be an internal procedure that is passed as an
    ! argument: gfortran implements that with trampolines on the stack, which
    ! force the shared library to request an executable stack (and crash when
    ! the loader refuses it). Instead the callbacks are module procedures that
    ! read the C function pointers and priors from these module variables.
    abstract interface
        function c_loglikelihood_t(theta,nDims,phi,nDerived) bind(c)
            import :: c_int, c_double
            integer(c_int), intent(in), value :: nDims
            integer(c_int), intent(in), value :: nDerived
            real(c_double), intent(in),  dimension(nDims) :: theta
            real(c_double), intent(out),  dimension(nDerived) :: phi
            real(c_double) :: c_loglikelihood_t
        end function c_loglikelihood_t
        subroutine c_prior_t(cube,theta,nDims) bind(c)
            import :: c_int, c_double
            integer(c_int), intent(in), value :: nDims
            real(c_double), intent(in),  dimension(nDims) :: cube
            real(c_double), intent(out), dimension(nDims) :: theta
        end subroutine c_prior_t
        subroutine c_dumper_t(ndead, nlive, npars, live, dead, logweights, logZ, logZerr) bind(c)
            import :: c_int, c_double
            integer(c_int), intent(in), value :: ndead, nlive, npars
            real(c_double), intent(in), dimension(npars,nlive) :: live, dead(npars,ndead), logweights(ndead)
            real(c_double), intent(in), value :: logZ, logZerr
        end subroutine c_dumper_t
        subroutine c_setup_loglikelihood_t() bind(c)
        end subroutine c_setup_loglikelihood_t
    end interface

    procedure(c_loglikelihood_t), pointer, save :: f_loglikelihood_ptr => null()
    procedure(c_prior_t), pointer, save         :: f_prior_ptr => null()
    procedure(c_dumper_t), pointer, save        :: f_dumper_ptr => null()
    procedure(c_setup_loglikelihood_t), pointer, save :: f_setup_loglikelihood_ptr => null()
    type(prior), dimension(:), allocatable, save :: module_priors
    interface run_polychord
        module procedure run_polychord_full, run_polychord_no_prior, run_polychord_no_dumper, run_polychord_no_prior_no_dumper, run_polychord_ini
    end interface run_polychord

contains


    subroutine run_polychord_full(loglikelihood, prior_transform, dumper, settings_in)
        use settings_module,          only: program_settings,initialise_settings
        use random_module,            only: initialise_random
        use nested_sampling_module,   only: NestedSampling
#ifdef MPI
        use mpi_module,               only: initialise_mpi, finalise_mpi
#endif
        implicit none
#ifdef MPI
        include 'mpif.h'
#endif

        interface
            function loglikelihood(theta,phi)
                import :: dp
                real(dp), intent(in),  dimension(:) :: theta
                real(dp), intent(out),  dimension(:) :: phi
                real(dp) :: loglikelihood
            end function loglikelihood
        end interface
        interface
            function prior_transform(cube) result(theta)
                import :: dp
                real(dp), intent(in), dimension(:) :: cube
                real(dp), dimension(size(cube))    :: theta
            end function prior_transform
        end interface
        interface
            subroutine dumper(live, dead, logweights, logZ, logZerr)
                import :: dp
                real(dp), intent(in) :: live(:,:), dead(:,:), logweights(:)
                real(dp), intent(in) :: logZ, logZerr
            end subroutine dumper
        end interface

        type(program_settings),intent(in)    :: settings_in
        type(program_settings)               :: settings 

        real(dp), dimension(4) :: output_info

#ifdef MPI
        call initialise_mpi(settings_in%feedback)
#endif
        if (settings_in%seed >= 0) then
            call initialise_random(settings_in%seed)
        else
            call initialise_random()
        end if
        settings = settings_in
        call initialise_settings(settings)   
#ifdef MPI
        output_info = NestedSampling(loglikelihood,prior_transform,dumper,settings,MPI_COMM_WORLD) 
        call finalise_mpi
#else
        output_info = NestedSampling(loglikelihood,prior_transform,dumper,settings,0) 
#endif

    end subroutine run_polychord_full


    !===================== INTERFACE ===============================================
    subroutine run_polychord_no_prior(loglikelihood,dumper,settings)
        use settings_module,          only: program_settings
        implicit none
        interface
            function loglikelihood(theta,phi)
                import :: dp
                real(dp), intent(in),  dimension(:) :: theta
                real(dp), intent(out),  dimension(:) :: phi
                real(dp) :: loglikelihood
            end function loglikelihood
        end interface
        interface
            subroutine dumper(live, dead, logweights, logZ, logZerr)
                import :: dp
                real(dp), intent(in) :: live(:,:), dead(:,:), logweights(:)
                real(dp), intent(in) :: logZ, logZerr
            end subroutine dumper
        end interface
        type(program_settings),intent(in)    :: settings  ! The program settings 
        call run_polychord(loglikelihood,identity_prior_transform,dumper,settings)
    end subroutine run_polychord_no_prior

    function identity_prior_transform(cube) result(theta)
        implicit none
        real(dp), intent(in), dimension(:) :: cube
        real(dp), dimension(size(cube))    :: theta
        theta = cube
    end function identity_prior_transform

    subroutine null_dumper(live, dead, logweights, logZ, logZerr)
        real(dp), intent(in) :: live(:,:), dead(:,:), logweights(:)
        real(dp), intent(in) :: logZ, logZerr
    end subroutine null_dumper

    subroutine run_polychord_no_dumper(loglikelihood,prior_transform,settings)
        use settings_module,          only: program_settings
        implicit none
        interface
            function loglikelihood(theta,phi)
                import :: dp
                real(dp), intent(in),  dimension(:) :: theta
                real(dp), intent(out),  dimension(:) :: phi
                real(dp) :: loglikelihood
            end function loglikelihood
        end interface
        interface
            function prior_transform(cube) result(theta)
                import :: dp
                real(dp), intent(in), dimension(:) :: cube
                real(dp), dimension(size(cube))    :: theta
            end function prior_transform
        end interface
        type(program_settings),intent(in)    :: settings  ! The program settings 
        call run_polychord(loglikelihood,prior_transform,null_dumper,settings)
    end subroutine run_polychord_no_dumper

    subroutine run_polychord_no_prior_no_dumper(loglikelihood, settings)
        use settings_module,          only: program_settings
        implicit none
        interface
            function loglikelihood(theta,phi)
                import :: dp
                real(dp), intent(in),  dimension(:) :: theta
                real(dp), intent(out),  dimension(:) :: phi
                real(dp) :: loglikelihood
            end function loglikelihood
        end interface
        type(program_settings),intent(in)    :: settings  ! The program settings 
        call run_polychord(loglikelihood,identity_prior_transform,null_dumper,settings)
    end subroutine run_polychord_no_prior_no_dumper


    subroutine run_polychord_ini(loglikelihood, setup_loglikelihood, input_file)
        use ini_module,               only: read_params
        use params_module,            only: add_parameter,param_type
        use priors_module,            only: create_priors
        use settings_module,          only: program_settings
        use utils_module,             only: STR_LENGTH
        implicit none
        type(program_settings)                    :: settings  ! The program settings 

        character(len=STR_LENGTH), intent(in)     :: input_file     ! input file
        type(param_type),dimension(:),allocatable :: params         ! Parameter array
        type(param_type),dimension(:),allocatable :: derived_params ! Derived parameter array
        interface
            function loglikelihood(theta,phi)
                import :: dp
                real(dp), intent(in),  dimension(:) :: theta
                real(dp), intent(out),  dimension(:) :: phi
                real(dp) :: loglikelihood
            end function loglikelihood
        end interface
        interface
            subroutine setup_loglikelihood(settings)
                import :: program_settings
                type(program_settings), intent(in) :: settings
            end subroutine setup_loglikelihood
        end interface

        call read_params(trim(input_file),settings,params,derived_params)
        call create_priors(module_priors,params,settings)
        call setup_loglikelihood(settings)
        call run_polychord(loglikelihood,prior_wrapper,settings) 
    end subroutine run_polychord_ini

    function prior_wrapper(cube) result(theta)
        implicit none
        real(dp), intent(in), dimension(:) :: cube
        real(dp), dimension(size(cube))    :: theta
        theta = hypercube_to_physical(cube,module_priors)
    end function prior_wrapper


    subroutine polychord_c_interface(&
            c_loglikelihood_ptr,&
            c_prior_ptr,&
            c_dumper_ptr,&
            nlive,&
            num_repeats,&
            nprior,&
            do_clustering,&
            feedback, &
            precision_criterion,&
            logzero,&
            max_ndead,&
            boost_posterior,&
            posteriors,&
            equals,&
            cluster_posteriors,&
            write_resume, &
            write_paramnames,&
            read_resume,&
            write_stats,&
            write_live,&
            write_dead,&
            write_prior,&
            compression_factor,&
            nDims,&
            nDerived, &
            base_dir,&
            file_root,&
            nGrade,&
            grade_frac,&
            grade_dims,&
            n_nlives,&
            loglikes,&
            nlives,&
            seed) &
            bind(c,name='polychord_c_interface')

        use iso_c_binding
        use utils_module,             only: STR_LENGTH, convert_c_string
        use ini_module,               only: default_params
        use params_module,            only: param_type
        use settings_module,          only: program_settings,initialise_settings
        use random_module,            only: initialise_random
        use nested_sampling_module,   only: NestedSampling
        use read_write_module,        only: write_paramnames_file

        ! ~~~~~~~ Local Variable Declaration ~~~~~~~
        implicit none

        interface
            function c_loglikelihood(theta,nDims,phi,nDerived) bind(c)
                use iso_c_binding
                integer(c_int), intent(in), value :: nDims
                integer(c_int), intent(in), value :: nDerived
                real(c_double), intent(in),  dimension(nDims) :: theta
                real(c_double), intent(out),  dimension(nDerived) :: phi
                real(c_double) :: c_loglikelihood
            end function c_loglikelihood
        end interface
        interface
            subroutine c_prior(cube,theta,nDims) bind(c)
                use iso_c_binding
                integer(c_int), intent(in), value :: nDims
                real(c_double), intent(in),  dimension(nDims) :: cube
                real(c_double), intent(out), dimension(nDims) :: theta
            end subroutine c_prior
        end interface
        interface
            subroutine c_dumper(ndead, nlive, npars, live, dead, logweights, logZ, logZerr) bind(c)
                use iso_c_binding
                integer(c_int), intent(in), value :: ndead, nlive, npars
                real(c_double), intent(in), dimension(npars,nlive) :: live, dead(npars,ndead), logweights(ndead)
                real(c_double), intent(in), value :: logZ, logZerr
            end subroutine c_dumper
        end interface

        type(c_funptr), intent(in), value   :: c_loglikelihood_ptr
        type(c_funptr), intent(in), value   :: c_prior_ptr
        type(c_funptr), intent(in), value   :: c_dumper_ptr
        integer(c_int), intent(in), value   :: nlive
        integer(c_int), intent(in), value   :: num_repeats
        integer(c_int), intent(in), value   :: nprior
        logical(c_bool), intent(in), value  :: do_clustering
        integer(c_int), intent(in), value   :: feedback
        real(c_double), intent(in), value   :: precision_criterion
        real(c_double), intent(in), value   :: logzero
        integer(c_int), intent(in), value   :: max_ndead
        real(c_double), intent(in), value   :: boost_posterior
        logical(c_bool), intent(in), value  :: posteriors
        logical(c_bool), intent(in), value  :: equals
        logical(c_bool), intent(in), value  :: cluster_posteriors
        logical(c_bool), intent(in), value  :: write_resume
        logical(c_bool), intent(in), value  :: write_paramnames
        logical(c_bool), intent(in), value  :: read_resume
        logical(c_bool), intent(in), value  :: write_stats
        logical(c_bool), intent(in), value  :: write_live
        logical(c_bool), intent(in), value  :: write_dead
        logical(c_bool), intent(in), value  :: write_prior
        real(c_double),  intent(in), value  :: compression_factor
        integer(c_int), intent(in), value   :: nDims
        integer(c_int), intent(in), value   :: nDerived
        character(len=1,kind=c_char), intent(in), dimension(STR_LENGTH) :: base_dir
        character(len=1,kind=c_char), intent(in), dimension(STR_LENGTH) :: file_root

        integer(c_int), intent(in), value             :: nGrade
        real(c_double), intent(in), dimension(nGrade) :: grade_frac
        integer(c_int), intent(in), dimension(nGrade) :: grade_dims

        integer(c_int), intent(in), value               :: n_nlives
        real(c_double), intent(in), dimension(n_nlives) :: loglikes
        integer(c_int), intent(in), dimension(n_nlives) :: nlives
        integer(c_int), intent(in), value               :: seed

        type(program_settings)    :: settings  ! The program settings 

        type(param_type),dimension(:),allocatable :: params         ! Parameter array
        type(param_type),dimension(:),allocatable :: derived_params ! Derived parameter array

        settings%nlive               = nlive                
        settings%num_repeats         = num_repeats          
        settings%nprior              = nprior
        settings%do_clustering       = do_clustering        
        settings%feedback            = feedback             
        settings%precision_criterion = precision_criterion  
        settings%logzero             = logzero  
        settings%max_ndead           = max_ndead            
        settings%boost_posterior     = boost_posterior      
        settings%posteriors          = posteriors           
        settings%equals              = equals               
        settings%cluster_posteriors  = cluster_posteriors   
        settings%write_resume        = write_resume         
        settings%write_paramnames    = write_paramnames     
        settings%read_resume         = read_resume          
        settings%write_stats         = write_stats          
        settings%write_live          = write_live           
        settings%write_dead          = write_dead           
        settings%write_prior         = write_prior
        settings%compression_factor  = compression_factor         
        settings%nDims               = nDims
        settings%nDerived            = nDerived

        settings%base_dir            = convert_c_string(base_dir)
        settings%file_root           = convert_c_string(file_root) 

        settings%seed                = seed

        allocate(settings%grade_frac(nGrade),settings%grade_dims(nGrade))
        settings%grade_frac = grade_frac
        settings%grade_dims = grade_dims

        allocate(settings%loglikes(n_nlives),settings%nlives(n_nlives))
        settings%loglikes = loglikes
        settings%nlives = nlives

        if(settings%write_paramnames) then
            params = default_params(settings%nDims,'theta','\theta')
            derived_params = default_params(settings%nDerived,'phi','\phi')
            call write_paramnames_file(settings,params,derived_params)
        end if

        call c_f_procpointer(c_loglikelihood_ptr, f_loglikelihood_ptr)
        call c_f_procpointer(c_prior_ptr, f_prior_ptr)
        call c_f_procpointer(c_dumper_ptr, f_dumper_ptr)

        call run_polychord(fort_loglikelihood,c_prior_transform,fort_dumper,settings) 

    end subroutine polychord_c_interface

        function fort_loglikelihood(theta,phi)
            implicit none
            real(dp), intent(in),  dimension(:) :: theta
            real(dp), intent(out),  dimension(:) :: phi
            real(dp) :: fort_loglikelihood

            real (c_double),dimension(size(theta)) :: c_theta
            integer (c_int)                        :: c_nDims
            real (c_double),dimension(size(phi))   :: c_phi
            integer (c_int)                        :: c_nDerived
            real (c_double)                        :: c_loglike

            c_nDims = size(theta)
            c_nDerived = size(phi)
            c_theta = theta
            c_loglike = f_loglikelihood_ptr(c_theta,c_nDims,c_phi,c_nDerived)
            phi = c_phi
            fort_loglikelihood = c_loglike

        end function fort_loglikelihood

        function c_prior_transform(cube) result(theta)
            implicit none
            real(dp), intent(in), dimension(:) :: cube
            real(dp), dimension(size(cube))    :: theta

            integer (c_int)                       :: c_nDims
            real (c_double),dimension(size(cube)) :: c_cube
            real (c_double),dimension(size(cube)) :: c_theta

            c_nDims = size(cube)
            c_cube = cube
            call f_prior_ptr(c_cube,c_theta,c_nDims)
            theta = c_theta

        end function c_prior_transform

        subroutine fort_dumper(live, dead, logweights, logZ, logZerr)
            implicit none
            real(dp), intent(in) :: live(:,:), dead(:,:), logweights(:)
            real(dp), intent(in) :: logZ, logZerr

            integer(c_int) :: c_ndead, c_nlive, c_npars
            real(c_double) :: c_live(size(live,1),size(live,2)), c_dead(size(dead,1),size(dead,2)), c_logweights(size(logweights))
            real(c_double) :: c_logZ, c_logZerr

            c_npars = size(live,1)
            c_nlive = size(live,2)
            c_ndead = size(dead,2)
            c_live = live
            c_dead = dead
            c_logweights = logweights
            c_logZ = logZ
            c_logZerr = logZerr
            call f_dumper_ptr(c_ndead, c_nlive, c_npars, c_live, c_dead, c_logweights, c_logZ, c_logZerr)
        end subroutine fort_dumper


    subroutine polychord_c_interface_ini(c_loglikelihood_ptr, c_setup_loglikelihood_ptr, input_file_c)&
            bind(c,name='polychord_c_interface_ini')

        use iso_c_binding
        use utils_module,             only: STR_LENGTH, convert_c_string

        ! ~~~~~~~ Local Variable Declaration ~~~~~~~
        implicit none

        interface
            function c_loglikelihood(theta,nDims,phi,nDerived) bind(c)
                use iso_c_binding
                integer(c_int), intent(in), value :: nDims
                integer(c_int), intent(in), value :: nDerived
                real(c_double), intent(in),  dimension(nDims) :: theta
                real(c_double), intent(out),  dimension(nDerived) :: phi
                real(c_double) :: c_loglikelihood
            end function c_loglikelihood
        end interface
        interface
            subroutine c_setup_loglikelihood() bind(c)
            end subroutine c_setup_loglikelihood
        end interface
        character(len=STR_LENGTH)     :: input_file     ! input file

        type(c_funptr), intent(in), value   :: c_loglikelihood_ptr
        type(c_funptr), intent(in), value   :: c_setup_loglikelihood_ptr
        character(len=1,kind=c_char), intent(in), dimension(STR_LENGTH) :: input_file_c

        input_file = convert_c_string(input_file_c)

        call c_f_procpointer(c_loglikelihood_ptr, f_loglikelihood_ptr)
        call c_f_procpointer(c_setup_loglikelihood_ptr, f_setup_loglikelihood_ptr)

        call run_polychord(fort_loglikelihood_ini, c_setup_loglikelihood_wrapper, input_file) 

    end subroutine polychord_c_interface_ini

        function fort_loglikelihood_ini(theta,phi)
            implicit none
            real(dp), intent(in),  dimension(:) :: theta
            real(dp), intent(out),  dimension(:) :: phi
            real(dp) :: fort_loglikelihood_ini

            real (c_double),dimension(size(theta)) :: c_theta
            integer (c_int)                        :: c_nDims
            real (c_double),dimension(size(phi))   :: c_phi
            integer (c_int)                        :: c_nDerived
            real (c_double)                        :: c_loglike

            c_nDims = size(theta)
            c_nDerived = size(phi)
            c_theta = theta
            c_loglike = f_loglikelihood_ptr(c_theta,c_nDims,c_phi,c_nDerived)
            phi = c_phi
            fort_loglikelihood_ini = c_loglike

        end function fort_loglikelihood_ini

        subroutine c_setup_loglikelihood_wrapper(settings)
            use settings_module,          only: program_settings
            implicit none
            type(program_settings), intent(in)    :: settings
            call f_setup_loglikelihood_ptr()
        end subroutine c_setup_loglikelihood_wrapper


end module interfaces_module
