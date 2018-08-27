set (SOURCES
    CommandLineArguments.cpp
    CudaConfiguration.cu
    DynamicalVariables.cu
    EnergyMomentumTensor.cu
    EquationOfState.cu
    EulerStep.cu
    FileIO.cpp
    FiniteDifference.cu
    FluxFunctions.cu
    FluxLimiter.cu
    FullyDiscreteKurganovTadmorScheme.cu
    GhostCells.cu
    GlauberModel.cpp
    HalfSiteExtrapolation.cu
    HydrodynamicValidity.cu
    HydroParameters.cpp
    HydroPlugin.cpp
    HydroWrapper.cpp
    InitialConditionParameters.cpp
    InitialConditions.cpp
    LatticeParameters.cpp
    LocalPropagationSpeed.cu
    MonteCarloGlauberModel.cpp
    Properties.cpp
    RegulateDissipativeCurrents.cu
    RunHydroWrapper.cpp
    SemiDiscreteKurganovTadmorScheme.cu
    SourceTerms.cu
    SpectralRadius.cu
    TransportCoefficients.cu
    )

add_library(gpu-vh_lib SHARED ${SOURCES})
add_executable(gpu-vh ${SOURCES})


target_link_libraries (gpu-vh_lib ${GSL_LIBRARIES})
target_link_libraries (gpu-vh ${GSL_LIBRARIES})

#target_link_libraries (gpu-vh_lib ${GSL_LIBRARIES} ${Config_LIBRARIES})
#target_link_libraries (gpu-vh ${GSL_LIBRARIES} ${Config_LIBRARIES})

install(TARGETS gpu-vh DESTINATION ${CMAKE_HOME_DIRECTORY})
