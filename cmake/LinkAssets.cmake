# Points DST at the scene assets in SRC with a symbolic link instead of a copy (the assets are tens of GB).
# Usage: cmake -DSRC=<source assets dir> -DDST=<link path> -P LinkAssets.cmake
# - No SRC (a fresh clone has no assets/): nothing to link; the build goes on without bundled scene assets.
# - DST is already a link to SRC: nothing to do.
# - DST is a link elsewhere: it is replaced (only the link is removed, never what it points to).
# - DST is a real directory (a copy from an older build): it is left in place with a warning. Deleting it here would
#   remove gigabytes without a trace; move it to the Trash once and the next build links.
if(NOT DEFINED SRC OR NOT DEFINED DST)
    message(FATAL_ERROR "LinkAssets.cmake needs -DSRC=<dir> and -DDST=<link>")
endif()
if(NOT EXISTS "${SRC}")
    return()
endif()
if(IS_SYMLINK "${DST}")
    file(READ_SYMLINK "${DST}" CURRENT_TARGET)
    if(CURRENT_TARGET STREQUAL "${SRC}")
        return()
    endif()
    file(REMOVE "${DST}")
elseif(EXISTS "${DST}")
    message(WARNING "${DST} is a copied assets directory from an older build; it is kept and not linked. "
                    "Move it to the Trash (do not delete it with rm) and rebuild to link ${SRC} instead.")
    return()
endif()
get_filename_component(DST_PARENT "${DST}" DIRECTORY)
file(MAKE_DIRECTORY "${DST_PARENT}")
file(CREATE_LINK "${SRC}" "${DST}" SYMBOLIC)
