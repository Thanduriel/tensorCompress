cmake_minimum_required (VERSION 3.12)
project (tensorComp)

#---------------------------------------------------------------------------------------
# main project
#---------------------------------------------------------------------------------------

# find all source and header files
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.hpp" "src/*.inl")

# organize the files according to the organisation in the file system
# from: http:#cmake.3232098.n2.nabble.com/Keep-folder-structure-in-Visual-Studio-Project-generated-with-CMake-td7586044.html
macro(GroupSources curdir)
   file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir} ${PROJECT_SOURCE_DIR}/${curdir}/*)
   foreach(child ${children})
          if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
                  GroupSources(${curdir}/${child})
          else()
                  string(REPLACE "/" "\\" groupname ${curdir})
            source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
          endif()
   endforeach()
endmacro()

# execute tha macro
GroupSources(src)

add_executable(tensorComp ${SOURCE_FILES})
set_target_properties(tensorComp PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT tensorComp)

target_compile_options(tensorComp PRIVATE /wd4996)
target_compile_options(tensorComp PRIVATE /wd4005)
target_compile_options(tensorComp PRIVATE /openmp)

#---------------------------------------------------------------------------------------
# libraries
#---------------------------------------------------------------------------------------

find_path( AVCODEC_INCLUDE_DIR libavcodec/avcodec.h )
find_library( AVCODEC_LIBRARY avcodec )
target_include_directories( tensorComp PRIVATE ${AVCODEC_INCLUDE_DIR} )
target_link_libraries( tensorComp PRIVATE ${AVCODEC_LIBRARY})

#find_path( AVFORMAT_INCLUDE_DIR libavformat/avformat.h )
find_library( AVFORMAT_LIBRARY avformat )
#target_include_directories( tensorComp PRIVATE ${AVFORMAT_INCLUDE_DIR} )
target_link_libraries( tensorComp PRIVATE ${AVFORMAT_LIBRARY})

find_library( AVUTILS_LIBRARY avutil )
target_link_libraries( tensorComp PRIVATE ${AVUTILS_LIBRARY})

find_library( SWSCALE_LIBRARY swscale )
target_link_libraries( tensorComp PRIVATE ${SWSCALE_LIBRARY})

target_include_directories(tensorComp PRIVATE "dependencies/eigen")
target_include_directories(tensorComp PRIVATE "dependencies/stb")
target_include_directories(tensorComp PRIVATE "src")