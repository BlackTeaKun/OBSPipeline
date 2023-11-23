cmake_minimum_required(VERSION 3.18)
project(helper)
set (TOP_DIR ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-march=native -DNDEBUG")
set(CMAKE_PREFIX_PATH "/home/zhangzirui/CMakeConfigs")
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
find_package(TBB REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
pkg_check_modules(HEALPIX REQUIRED IMPORTED_TARGET healpix_cxx)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${TOP_DIR}" CACHE PATH "${TOP_DIR}" FORCE)
endif()


add_library(${CMAKE_PROJECT_NAME} SHARED
    ./src/Driver.cxx
    ./src/PyDriver.cxx
    ./src/MapMaker.cxx
    ./src/reckon_impl.cxx
    ./src/sampling_impl.cxx
    ./src/addleakage_impl.cxx
    ./src/addleakage_bypass_impl.cxx
    ./src/Fitter.cxx
)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES SUFFIX ".so")
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME "${MODULENAME}")
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)

message("${TBB_INCLUDE}")
target_link_libraries( ${CMAKE_PROJECT_NAME} PRIVATE
    Python3::Python
    Python3::NumPy
    # ${TBB_IMPORTED_TARGETS}
    TBB::tbb
    PkgConfig::HEALPIX
)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ./src
    ${TBB_INCLUDE}
    # ${OpenMP_INCLUDE}
)
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
install (FILES $<TARGET_FILE:${CMAKE_PROJECT_NAME}> DESTINATION . )
