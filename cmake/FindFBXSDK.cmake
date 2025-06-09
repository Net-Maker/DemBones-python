find_path(FBXSDK_INCLUDE_DIR 
	NAMES "fbxsdk.h"
	PATHS 
		"${CMAKE_CURRENT_SOURCE_DIR}/ExtLibs/FBXSDK/include"
)

message(STATUS "FBXSDK_INCLUDE_DIR: ${FBXSDK_INCLUDE_DIR}")

if(WIN32)
	find_path(FBXSDK_LIBS_DIR
		NAMES "libfbxsdk-md.lib"
		PATHS
			"${CMAKE_CURRENT_SOURCE_DIR}/ExtLibs/FBXSDK/lib/vs2017/x64/release"
			"${CMAKE_CURRENT_SOURCE_DIR}/ExtLibs/FBXSDK/lib/vs2015/x64/release"
	)
	file(GLOB FBXSDK_LIBS "${FBXSDK_LIBS_DIR}/*-md.lib")
else()
	set(FBXSDK_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ExtLibs/FBXSDK/lib/release")
	message(STATUS "FBXSDK_LIBS_DIR: ${FBXSDK_LIBS_DIR}")
	
	set(FBXSDK_LIBS "${FBXSDK_LIBS_DIR}/libfbxsdk.a")
	
	
	message(STATUS "FBXSDK_LIBS set to: ${FBXSDK_LIBS}")
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  FBXSDK DEFAULT_MSG FBXSDK_INCLUDE_DIR FBXSDK_LIBS
)
