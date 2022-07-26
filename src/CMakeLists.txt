cmake_minimum_required(VERSION 3.0.0)

set (OVERSETGRID 1)

if(OVERSETGRID STREQUAL 1) 

    message(STATUS "Simulation with overset grid")
    project(LBM-OversetGrid)

    set(LIB_NAME cudaLib)
    add_subdirectory(${LIB_NAME} ${CMAKE_SOURCE_DIR}/build/${PROJECT_NAME}/cudaLibBuild)
    set(EXTRA_LIBS ${EXTRA_LIBS} CudaKernel)

    include_directories(
        ${CMAKE_SOURCE_DIR}/../3rdLibs
        ${CMAKE_SOURCE_DIR}/${LIB_NAME}/basic
        ${CMAKE_SOURCE_DIR}/${LIB_NAME}/kernelsOverset
        ${CMAKE_SOURCE_DIR}/src
    )

    file(GLOB_RECURSE SRC_FILES 
        ${CMAKE_SOURCE_DIR}/test/testOversetGrid.cpp
        ${CMAKE_SOURCE_DIR}/src/*.hpp
        ${CMAKE_SOURCE_DIR}/src/*.cpp
    )

    ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_FILES})
    target_link_libraries(${PROJECT_NAME} ${EXTRA_LIBS})

elseif(CHANNELFLOW STREQUAL 1)

    message(STATUS "Simulation of channel flow")
    project(LBM-ChannelFlow)

    set(LIB_NAME cudaLib)
    add_subdirectory(${LIB_NAME} ${CMAKE_SOURCE_DIR}/build/${PROJECT_NAME}/cudaLibBuild)
    set(EXTRA_LIBS ${EXTRA_LIBS} CudaKernel)

    include_directories(
        ${CMAKE_SOURCE_DIR}/../3rdLibs
        ${CMAKE_SOURCE_DIR}/${LIB_NAME}
        ${CMAKE_SOURCE_DIR}/src
    )

    file(GLOB_RECURSE SRC_FILES 
        ${CMAKE_SOURCE_DIR}/test/testChannelFlow.cpp
        ${CMAKE_SOURCE_DIR}/src/*.hpp
        ${CMAKE_SOURCE_DIR}/src/*.cpp
    )

    ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_FILES})
    target_link_libraries(${PROJECT_NAME} ${EXTRA_LIBS})
else()
    message(STATUS "Standard LBM simulation")
    project(LBM)

    set(LIB_NAME cudaLib)
    add_subdirectory(${LIB_NAME} ${CMAKE_SOURCE_DIR}/build/${PROJECT_NAME}/cudaLibBuild)
    set(EXTRA_LIBS ${EXTRA_LIBS} CudaKernel)

    include_directories(
        ${CMAKE_SOURCE_DIR}/../3rdLibs
        ${CMAKE_SOURCE_DIR}/${LIB_NAME}
        ${CMAKE_SOURCE_DIR}/src
    )

    file(GLOB_RECURSE SRC_FILES 
        ${CMAKE_SOURCE_DIR}/test/test.cpp
        ${CMAKE_SOURCE_DIR}/src/*.hpp
        ${CMAKE_SOURCE_DIR}/src/*.cpp
    )

    ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_FILES})
    target_link_libraries(${PROJECT_NAME} ${EXTRA_LIBS})
endif()



