#ifndef MEMORY_STATE_H
#define MEMORY_STATE_H

#include <iostream>
#include <thread>
#include <chrono>
#include <string.h>
#include <unistd.h>      
#include <sys/sysinfo.h> 
#include <sys/vfs.h>  
#include <string> 


void UseCondition();
float getGPUUsage();      
int getGPUFrequency();   

#endif