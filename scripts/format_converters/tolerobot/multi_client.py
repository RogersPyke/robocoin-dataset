import argparse
import asyncio
import logging
import multiprocessing as mp
import time  # 用于判断执行时长

from robocoin_dataset.format_converter.tolerobot.client import LeFormatConverterTaskClient
from robocoin_dataset.utils.logger import setup_logger

# 执行同步任务，并返回执行耗时
def client_run_sync(client):
    start = time.time()
    asyncio.run(client.run())
    cost = time.time() - start
    return cost  # 返回任务执行耗时

async def run_client_process(
    server_uri: str,
    heartbeat_interval: float,
    log_path: str,
    process_id: int,
    min_task_cost: float = 1.0,  # 大于这个时间，才算【真正执行了任务】
) -> None:
    logger = setup_logger(
        name=f"client_{process_id}",
        log_dir=log_path,
        level=logging.ERROR,
    )

    logger.info(f"✅ 客户端 {process_id} 已启动，准备接收任务")

    while True:
        try:
            # 1. 创建客户端，尝试连服务端拿任务
            client = LeFormatConverterTaskClient(
                server_uri=server_uri,
                heartbeat_interval=heartbeat_interval,
                logger=logger,
            )

            # 2. 执行任务，并获取耗时
            loop = asyncio.get_running_loop()
            task_cost = await loop.run_in_executor(None, client_run_sync, client)

            # 3. 通过执行时长判断：是否真的跑了任务
            if task_cost >= min_task_cost:
                logger.info(f"✅ 客户端 {process_id} 完成一个有效任务，耗时：{task_cost:.2f}s，继续取下一个...")
                # 回到循环开头，继续取下一个任务
                continue

            else:
                # 耗时极短 = 服务端没有任务分配
                logger.info(f"🛑 客户端 {process_id} 未获取到新任务，进程自动停止")
                break  # 退出循环 → 关闭客户端

        except Exception as e:
            logger.error(f"⚠️ 客户端 {process_id} 异常，5秒后重试: {str(e)}")
            await asyncio.sleep(5)


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

    print("✅ 所有客户端已安全退出")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
