@echo off
chcp 65001 >nul
title Local LLM Server (Qwen3.5-35B)

cd /d "D:\llama_cpp_latest\llama-b8575"

echo ========================================================
echo  Starting Local LLM Server...
echo  Model : Qwen3.5-35B-A3B-Q4_K_M
echo  GPU   : -ngl 99, tensor-split 7:3
echo  Port  : 8080
echo  Cache : --cache-reuse 256
echo ========================================================
echo.
echo  Start main.py after this window shows "HTTP server listening".
echo  The app will auto-detect this server and connect directly.
echo  Close this window to stop the LLM server.
echo ========================================================

set "PATH=C:\Users\Lucas\.conda\envs\xtts-env\Lib\site-packages\torch\lib;%PATH%"
:: nvidia-smi index 1 = RTX 4070 Ti SUPER (eGPU, 16GB)
:: LLM runs on Ti SUPER + CPU overflow; TTS runs on Laptop GPU (cuda:1, internal PCIe) -- no contention
llama-server.exe ^
  -m "D:\llama_cpp_latest\llama.cpp\Qwen3.5-35B-A3B-Q4_K_M.gguf" ^
  --no-mmap ^
  -ngl 0 ^
  -c 4096 ^
  -t 12 ^
  --ubatch-size 512 ^
  --batch-size 2048 ^
  --cache-reuse 256 ^
  --reasoning-budget 0 ^
  --chat-template-kwargs "{\"enable_thinking\": false}" ^
  --port 8080 ^
  -a qwen3.5-35b-a3b

echo.
echo Server stopped.
pause
