#include <lofar_config.h>
#include <MACIO/MACServiceInfo.h>

#include <iostream>

int main()
{
  std::cout << "production: " << "ccu001:" << CEP_LOGPROC_LOGGING << std::endl;
  std::cout << "test: " << "ccu099:" << CEP_LOGPROC_LOGGING << std::endl;
}
