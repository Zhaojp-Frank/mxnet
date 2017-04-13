#include <iostream>
#include <mpi.h>
#include <thread>

using namespace std::chrono;
void DoTOFU(int rank, int size, long buf_size, int level, int iterations, int sleep_duration) {
  char* buf_send = new char[buf_size * level * size];
  char* buf_recv = new char[buf_size * level * size];
  MPI_Request sends[8192], recvs[8192];
  bool send_dones[8192];
  bool recv_dones[8192];

  //MPI_Barrier(MPI_COMM_WORLD);

  auto begin = high_resolution_clock::now();
  auto begin_ms = duration_cast<milliseconds>(begin.time_since_epoch()).count();

  for (int i = 0; i < iterations; i++) {
    for (int l = 0; l < level; l++) {
      for (int j = 0; j < size; j++) {
        if (j == rank) {
          continue;
        }
        int idx = j * level + l;
        MPI_Irecv(&buf_recv[idx * buf_size], (int)buf_size,  MPI_BYTE, j, l, MPI_COMM_WORLD, &recvs[idx]);
        recv_dones[idx] = false;
      }
    }
    for (int l = 0; l < level; l++) {
      for (int j = 0; j < size; j++) {
        if (j == rank) {
          continue;
        }
        int idx = j * level + l;
        MPI_Isend(&buf_send[idx * buf_size], (int)buf_size,  MPI_BYTE, j, l, MPI_COMM_WORLD, &sends[idx]);
        send_dones[idx] = false;
      }
      int send_count = size - 1;
      int recv_count = size - 1;
      while (send_count > 0 || recv_count > 0) {
        do {
        for (int k = 0; k < level; k++) {
          for (int j = 0; j < size; j++) {
            if (j == rank) {
              continue;
            }
            int idx = j * level + k;
            if (!recv_dones[idx]) {
              int flag;
              MPI_Test(&recvs[idx], &flag, MPI_STATUS_IGNORE);
              if (flag) {
                MPI_Wait(&recvs[idx], MPI_STATUS_IGNORE);
                recv_count--;
                recv_dones[idx] = true;
              }
            }
          }
        }
        for (int j = 0; j < size; j++) {
          if (j == rank) {
            continue;
          }
          int idx = j * level + l;
          if (!send_dones[idx]) {
            int flag;
            MPI_Test(&sends[idx], &flag, MPI_STATUS_IGNORE);
            if (flag) {
              MPI_Wait(&sends[idx], MPI_STATUS_IGNORE);
              send_count--;
              send_dones[idx] = true;
            }
          }
        }
        } while(send_count > 0) ;
        std::this_thread::sleep_for(milliseconds(sleep_duration));
      }
    }
    //MPI_Barrier(MPI_COMM_WORLD);
  }

  auto now = high_resolution_clock::now();
  auto now_ms = duration_cast<milliseconds>(now.time_since_epoch()).count();
  float rate = buf_size * 8.0 * (size - 1) * size * level * iterations / 
                    ((now_ms - begin_ms) / 1000.0) / 1024.0 / 1024.0 / 1024.0;
  std::cout << rate << std::endl;
}


void DoDP(int rank, int size, long buf_size, int level, int iterations, int sleep_duration) {
  char* buf_send = new char[buf_size * level * size];
  char* buf_recv = new char[buf_size * level * size];
  MPI_Request sends[8192], recvs[8192];
  bool send_dones[8192];
  bool recv_dones[8192];

  //MPI_Barrier(MPI_COMM_WORLD);

  auto begin = high_resolution_clock::now();
  auto begin_ms = duration_cast<milliseconds>(begin.time_since_epoch()).count();

  for (int i = 0; i < iterations; i++) {
    for (int l = 0; l < level; l++) {
      for (int j = 0; j < size; j++) {
        if (j == rank) {
          continue;
        }
        int idx = j * level + l;
        MPI_Irecv(&buf_recv[idx * buf_size], (int)buf_size,  MPI_BYTE, j, l, MPI_COMM_WORLD, &recvs[idx]);
        recv_dones[idx] = false;
        MPI_Isend(&buf_send[idx * buf_size], (int)buf_size,  MPI_BYTE, j, l, MPI_COMM_WORLD, &sends[idx]);
        send_dones[idx] = false;
      }
    }
    int send_count = (size - 1) * level;
    int recv_count = (size - 1) * level;
    while (send_count > 0 || recv_count > 0) {
      for (int l = 0; l < level; l++) {
        for (int j = 0; j < size; j++) {
          if (j == rank) {
            continue;
          }
          int idx = j * level + l;
          if (!send_dones[idx]) {
            int flag;
            MPI_Test(&sends[idx], &flag, MPI_STATUS_IGNORE);
            if (flag) {
              MPI_Wait(&sends[idx], MPI_STATUS_IGNORE);
              send_count--;
              send_dones[idx] = true;
            }
          }
          if (!recv_dones[idx]) {
            int flag;
            MPI_Test(&recvs[idx], &flag, MPI_STATUS_IGNORE);
            if (flag) {
              MPI_Wait(&recvs[idx], MPI_STATUS_IGNORE);
              recv_count--;
              recv_dones[idx] = true;
            }
          }
        }
      }
      std::this_thread::sleep_for(milliseconds(sleep_duration));
    }
    //MPI_Barrier(MPI_COMM_WORLD);
  }

  auto now = high_resolution_clock::now();
  auto now_ms = duration_cast<milliseconds>(now.time_since_epoch()).count();
  float rate = buf_size * 8.0 * (size - 1) * size * level * iterations / 
                    ((now_ms - begin_ms) / 1000.0) / 1024.0 / 1024.0 / 1024.0;
  std::cout << rate << std::endl;
}

int main(int argc, char**argv) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (atoi(argv[5]) == 0) {
    std::cout << "TOFU" << std::endl;
    DoTOFU(rank, size, atol(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
  } else {
    std::cout << "DP" << std::endl;
    DoDP(rank, size, atol(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
  }
  MPI_Finalize();
}
