cmake_minimum_required(VERSION 3.5)
project(hifivoice)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pie -fPIE -fPIC -Wall -O3")

find_package(ncnn REQUIRED)
if (ncnn_FOUND)
	message("NCNN FOUND")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ncnn_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ncnn_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ncnn_EXE_LINKER_FLAGS}")
    message(STATUS "ncnn_LIBS: ${ncnn_LIBS}")
    message(STATUS "ncnn_INCLUDE_DIRS: ${ncnn_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "ncnn Not Found!")
endif ()

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message("OpenCV FOUND")
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "opencv Not Found!")
endif (OpenCV_FOUND)

find_package(SndFile REQUIRED)
if (SndFile_FOUND)
    message("SndFile FOUND")
    message(STATUS "SndFile_LIBS: ${SNDFILE_LIBRARIES}")
    message(STATUS "SndFile_INCLUDE_DIRS: ${SNDFILE_INCLUDE_DIR}")
else ()
    message(FATAL_ERROR "SndFile Not Found!")
endif (SndFile_FOUND)

# openmp
option(hifivoice_OPENMP "openmp support" ON)
set(USE_OPEN_MP TRUE CACHE BOOL "Set to FALSE to not use OpenMP")
if (USE_OPEN_MP)
	find_package(OpenMP REQUIRED)
	if (OpenMP_FOUND)
		message("OpenMP FOUND")
		set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
		#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -pthread -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=4")
	else ()
    	message(FATAL_ERROR "openmp Not Found!")
	endif ()
endif ()

find_package(Vulkan REQUIRED)



include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ncnn/include/ncnn)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)
add_executable(hifivoice main.cpp hifivoice.cpp)
target_link_libraries( hifivoice ${SNDFILE_LIBRARIES} ncnn ${OpenCV_LIBS} ${Vulkan_LIBRARY})




