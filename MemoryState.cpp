#include "MemoryState.h"
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <iomanip> 

inline int GetCurrentPid() {
    return getpid(); 
}

#define PROCESS_ITEM 14

float getGPUUsage() {
    std::ifstream file("/sys/devices/platform/17000000.gpu/load");
    if (!file.is_open()) return 0.0f;

    std::string line;
    std::getline(file, line);
    file.close();

    try {
        return std::stof(line) / 10.0f;
    } catch (...) {
        return 0.0f;
    }
}

int getGPUFrequency() {
    std::ifstream file("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq");
    if (!file.is_open()) return 0;

    std::string line;
    std::getline(file, line);
    file.close();

    try {   
        return std::stoi(line) / 1000000; 
    } catch (...) {
        return 0;
    }
}

static const char* get_items(const char* buffer, unsigned int item) {
    const char* p = buffer;
    int len = strlen(buffer);
    int count = 0;

    for (int i = 0; i < len; i++) {
        if (' ' == *p) {
            count++;
            if (count == item - 1) {
                p++;
                break;
            }
        }
        p++;
    }
    return p;
}

static inline unsigned long get_cpu_total_occupy() {
    unsigned long user_time, nice_time, system_time, idle_time;
    FILE* fd;
    char buff[1024] = { 0 };

    fd = fopen("/proc/stat", "r");
    if (nullptr == fd) return 0;

    if (fgets(buff, sizeof(buff), fd)) {
        char name[64] = { 0 };
        sscanf(buff, "%s %ld %ld %ld %ld", name, &user_time, &nice_time, &system_time, &idle_time);
    }
    fclose(fd);

    return (user_time + nice_time + system_time + idle_time);
}

static inline unsigned long get_cpu_proc_occupy(int pid) {
    unsigned int tmp_pid;
    unsigned long utime, stime, cutime, cstime;
    char file_name[64] = { 0 };
    FILE* fd;
    char line_buff[1024] = { 0 };
    sprintf(file_name, "/proc/%d/stat", pid);

    fd = fopen(file_name, "r");
    if (nullptr == fd) return 0;

    if (fgets(line_buff, sizeof(line_buff), fd)) {
        sscanf(line_buff, "%u", &tmp_pid);
        const char* q = get_items(line_buff, PROCESS_ITEM);
        sscanf(q, "%ld %ld %ld %ld", &utime, &stime, &cutime, &cstime);
    }
    fclose(fd);

    return (utime + stime + cutime + cstime);
}

inline float GetCpuUsageRatio(int pid) {
    unsigned long totalcputime1, totalcputime2;
    unsigned long procputime1, procputime2;

    totalcputime1 = get_cpu_total_occupy();
    procputime1 = get_cpu_proc_occupy(pid);

    usleep(200000); // 采样间隔 200ms

    totalcputime2 = get_cpu_total_occupy();
    procputime2 = get_cpu_proc_occupy(pid);

    float pcpu = 0.0;
    if (0 != (totalcputime2 - totalcputime1)) {
        pcpu = (float)(procputime2 - procputime1) / (totalcputime2 - totalcputime1);
    }

    int cpu_num = get_nprocs(); 
    return pcpu * cpu_num;
}

inline float GetMemoryUsage(int pid) {
    char file_name[64] = { 0 };
    FILE* fd;
    char line_buff[512] = { 0 };
    sprintf(file_name, "/proc/%d/status", pid);
    
    fd = fopen(file_name, "r");
    if (nullptr == fd) return 0;

    char name[64];
    int vmrss = 0;
    while (fgets(line_buff, sizeof(line_buff), fd)) {
        if (strncmp(line_buff, "VmRSS:", 6) == 0) {
            sscanf(line_buff, "%s %d", name, &vmrss); 
            break;
        }
    }
    fclose(fd);
    
    return (float)vmrss / 1024.0; 
}

void UseCondition() {
    int current_pid = GetCurrentPid();
    float cpu_usage = GetCpuUsageRatio(current_pid) * 100.0f;
    float mem_usage = GetMemoryUsage(current_pid);
    float gpu_load = getGPUUsage();
    int gpu_freq = getGPUFrequency();

    std::cout << "------------------------------------------" << std::endl;
    std::cout << " [Jetson Orin Nano Super Monitor] PID: " << current_pid << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(1); // 设置小数点后1位
    std::cout << " CPU Usage: " << std::setw(5) << cpu_usage << " %" << std::endl;
    std::cout << " MEM Usage: " << std::setw(5) << mem_usage << " MB" << std::endl;
    std::cout << " GPU Load : " << std::setw(5) << gpu_load  << " %" << std::endl;
    std::cout << " GPU Freq : " << std::setw(5) << gpu_freq  << " MHz" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
}