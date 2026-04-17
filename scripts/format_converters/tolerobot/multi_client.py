import argparse
import asyncio
import logging
import multiprocessing as mp

from robocoin_dataset.format_converter.tolerobot.client import LeFormatConverterTaskClient
from robocoin_dataset.utils.logger import setup_logger

async def run_client_process(
    server_uri: str,
    heartbeat_interval: float,
    log_path: str,
    process_id: int,
    max_empty_tasks: int = 3,  # 连续3次无任务 → 退出
    empty_sleep_time: float = 2.0,  # 无任务时休眠2秒
) -> None:
    logger = setup_logger(
        name=f"client_{process_id}",
        log_dir=log_path,
        level=logging.INFO,  # 提升日志级别，方便调试
    )

    logger.info(f"✅ 客户端 {process_id} 已启动，任务处理模式:")
    logger.info(f"   - 有任务 → 立即处理")
    logger.info(f"   - 无任务 → 休眠{empty_sleep_time}秒，连续{max_empty_tasks}次无任务自动退出")

    empty_task_count = 0  # 无任务计数器
    
    while True:
        try:
            client = LeFormatConverterTaskClient(
                server_uri=server_uri,
                heartbeat_interval=heartbeat_interval,
                logger=logger,
            )

            # 执行任务
            result = await client.run()

            # ====================== 核心逻辑 ======================
            # 1. 判断是否真的执行了任务（通过返回结果判断）
            if result is None or (isinstance(result, dict) and result.get("converted_episodes", 0) == 0):
                empty_task_count += 1
                logger.info(f"🛑 无任务，连续无任务次数: {empty_task_count}/{max_empty_tasks}")
                
                # 2. 达到阈值 → 自动退出
                if empty_task_count >= max_empty_tasks:
                    logger.info(f"✅ 连续{max_empty_tasks}次无任务，客户端{process_id}自动退出")
                    break
                
                # 3. 未达阈值 → 休眠后重试
                await asyncio.sleep(empty_sleep_time)
                
            else:
                # 有任务执行 → 重置计数器，立即继续
                empty_task_count = 0
                logger.info(f"✅ 任务处理完成，继续接收下一个任务")
                await asyncio.sleep(0.1)  # 短暂休眠，避免CPU空转

        except Exception as e:
            # 异常处理：重置计数器，避免误判退出
            empty_task_count = 0
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
    # 🆕 新增参数：连续无任务次数阈值（默认3次）
    argparser.add_argument("--max-empty-tasks", type=int, default=3, 
                         help="连续无任务次数达到此值时自动退出（默认3）")

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
            )
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
