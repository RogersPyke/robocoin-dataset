import argparse
import asyncio
import logging
import multiprocessing as mp

from robocoin_dataset.format_converter.tolerobot.client import LeFormatConverterTaskClient
from robocoin_dataset.utils.logger import setup_logger

# 🔥 关键：用线程池执行阻塞任务，不卡住 asyncio
def client_run_sync(client):
    # 注意：这里不能 await，直接调用同步逻辑
    # 你的 client.run() 内部会执行阻塞 convert()
    asyncio.run(client.run())

async def run_client_process(
    server_uri: str,
    heartbeat_interval: float,
    log_path: str,
    process_id: int,
) -> None:
    logger = setup_logger(
        name=f"client_{process_id}",
        log_dir=log_path,
        level=logging.ERROR,
    )

    client = LeFormatConverterTaskClient(
        server_uri=server_uri,
        heartbeat_interval=heartbeat_interval,
        logger=logger,
    )

    # ✅ 核心修复：把整个 client 运行逻辑丢到线程池
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, client_run_sync, client)


def client_process_main(
    server_uri: str,
    heartbeat_interval: float,
    log_path: str,
    process_id: int,
) -> None:
    asyncio.run(
        run_client_process(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            log_path=log_path,
            process_id=process_id,
        )
    )


def main() -> None:
    argparser = argparse.ArgumentParser(description="Convert dataset to lerobot format.")
    argparser.add_argument("--host", type=str, default="127.0.0.1")
    argparser.add_argument("--port", type=int, default=8765)
    argparser.add_argument("--log-path", type=str, default="logs/client.log")
    argparser.add_argument("--timeout", type=float, default=1.0)
    argparser.add_argument("--heartbeat-interval", type=float, default=10.0)
    argparser.add_argument("--num-clients", type=int, default=1)

    args = argparser.parse_args()
    num_clients = max(1, min(args.num_clients, 8))
    server_uri = f"ws://{args.host}:{args.port}"

    processes = []
    for i in range(num_clients):
        proc = mp.Process(
            target=client_process_main,
            kwargs=dict(
                server_uri=server_uri,
                heartbeat_interval=args.heartbeat_interval,
                log_path=args.log_path,
                process_id=i,
            ),
        )
        proc.start()
        processes.append(proc)

    print(f"Started {num_clients} client processes. Waiting for them to finish...")

    try:
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        print("\nShutting down clients...")
        for proc in processes:
            proc.terminate()
            proc.join(timeout=2)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
