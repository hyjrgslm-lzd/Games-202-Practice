# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BINARY_7Z "OFF")
set(CPACK_BINARY_BUNDLE "")
set(CPACK_BINARY_CYGWIN "")
set(CPACK_BINARY_DEB "")
set(CPACK_BINARY_DRAGNDROP "")
set(CPACK_BINARY_FREEBSD "")
set(CPACK_BINARY_IFW "OFF")
set(CPACK_BINARY_NSIS "ON")
set(CPACK_BINARY_NUGET "OFF")
set(CPACK_BINARY_OSXX11 "")
set(CPACK_BINARY_PACKAGEMAKER "")
set(CPACK_BINARY_PRODUCTBUILD "")
set(CPACK_BINARY_RPM "")
set(CPACK_BINARY_STGZ "")
set(CPACK_BINARY_TBZ2 "")
set(CPACK_BINARY_TGZ "")
set(CPACK_BINARY_TXZ "")
set(CPACK_BINARY_TZ "")
set(CPACK_BINARY_WIX "OFF")
set(CPACK_BINARY_ZIP "OFF")
set(CPACK_BUILD_SOURCE_DIRS "F:/GAMES202/Assignment2/prt;F:/GAMES202/Assignment2/prt/build")
set(CPACK_CMAKE_GENERATOR "Visual Studio 16 2019")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_GENERATOR "NSIS")
set(CPACK_INSTALL_CMAKE_PROJECTS "F:/GAMES202/Assignment2/prt/build;nori;ALL;/")
set(CPACK_INSTALL_PREFIX "C:/Program Files (x86)/nori")
set(CPACK_MODULE_PATH "")
set(CPACK_NSIS_DISPLAY_NAME "nori 2.2.0")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
set(CPACK_NSIS_PACKAGE_NAME "nori 2.2.0")
set(CPACK_OUTPUT_CONFIG_FILE "F:/GAMES202/Assignment2/prt/build/CPackConfig.cmake")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "D:/cmake/share/cmake-3.15/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "nori built using CMake")
set(CPACK_PACKAGE_FILE_NAME "nori-2.2.0-win64")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "nori 2.2.0")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "nori 2.2.0")
set(CPACK_PACKAGE_NAME "nori")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "Humanity")
set(CPACK_PACKAGE_VERSION "2.2.0")
set(CPACK_PACKAGE_VERSION_MAJOR "2")
set(CPACK_PACKAGE_VERSION_MINOR "2")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_RESOURCE_FILE_LICENSE "D:/cmake/share/cmake-3.15/Templates/CPack.GenericLicense.txt")
set(CPACK_RESOURCE_FILE_README "D:/cmake/share/cmake-3.15/Templates/CPack.GenericDescription.txt")
set(CPACK_RESOURCE_FILE_WELCOME "D:/cmake/share/cmake-3.15/Templates/CPack.GenericWelcome.txt")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_7Z "")
set(CPACK_SOURCE_CYGWIN "")
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_SOURCE_IGNORE_FILES "/.git*;/.cvs*;")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "F:/GAMES202/Assignment2/prt/build/CPackSourceConfig.cmake")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "nori-2.2.0")
set(CPACK_SOURCE_RPM "")
set(CPACK_SOURCE_TBZ2 "")
set(CPACK_SOURCE_TGZ "")
set(CPACK_SOURCE_TXZ "")
set(CPACK_SOURCE_TZ "")
set(CPACK_SOURCE_ZIP "")
set(CPACK_SYSTEM_NAME "win64")
set(CPACK_TOPLEVEL_TAG "win64")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "F:/GAMES202/Assignment2/prt/build/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()
