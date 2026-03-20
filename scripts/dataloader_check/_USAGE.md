This stage is after visualize check.

to run this, use:
```
# Local mode
python scripts/dataloader_check/check.py local \
--config-path ./db/postgresql_config.yaml \
--log-dir ./logs/dataloader_check \
--num-workers 8 \
--sample-rate 0.1

# Server mode
python scripts/dataloader_check/check.py server \
--config-path ./db/postgresql_config.yaml \
--host 0.0.0.0 \
--port 2010 \
--log-dir ./logs/dataloader_check_server \
--num-workers 8 \
--sample-rate 0.1

# Client mode
python scripts/dataloader_check/check.py client \
--host localhost \
--port 2010 \
--log-dir ./logs/dataloader_check_client \
--num-clients 1 \
--heartbeat-interval 10.0
```