// stb_impl.cpp — Single translation unit that instantiates stb_image.
//
// STB_IMAGE_IMPLEMENTATION must be defined exactly once across the entire
// program. All other files that need stb_image just #include <stb_image.h>
// without the define.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include <stb_image.h>
