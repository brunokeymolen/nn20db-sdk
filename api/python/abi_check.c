/* nn20db-sdk
 *
 * ABI guard for the Python bindings: exposes the compiled size of
 * nn20db_config so nn20db.py can assert its ctypes mirror matches the
 * layout in the installed SDK headers at import time.
 */
#include <stddef.h>

#include "nn20db_config.h"

size_t nn20db_py_config_sizeof(void)
{
    return sizeof(nn20db_config);
}
