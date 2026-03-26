Before running this stage, first run `gen_readme.py` to generate README files.

This stage is designed to do upload only. It:
- never checks what to upload
- never generates anything
- just receives the path and does upload

There may be "backup" folders that are redundant, so a hook skips uploading them.

To use this script:

**HuggingFace**:



```bash
# Set a valid token.
# NEVER SAVE THE TOKEN HERE since this can trigger GIT err blocking commit and push!!!
export HF_TOKEN=YOUR_TOKEN_HERE

# Local mode (single machine)
python scripts/hub_upload/upload2hub.py \
--local \
--config hf.yaml

# Server mode (distribute tasks to clients) in tmux (recommended)
tmux new-session -d -s hf-upload-server "\
cd /home/rogerspyke/projects/robocoin-dataset && \
python scripts/hub_upload/upload2hub.py \
--server\
--config hf.yaml \
--host 0.0.0.0 \
--port 2100"
tmux attach -t hf-upload-server

# Client mode (connect to server and process tasks) in tmux (recommended)
tmux new-session -d -s hf-upload-client "\
cd /home/rogerspyke/projects/robocoin-dataset && \
python scripts/hub_upload/upload2hub.py \
 --client \
 --config hf.yaml \
 --host 127.0.0.1 \
 --port 2100 \
 --num-clients 4"
tmux attach -t hf-upload-client
```

**ModelScope**:

```bash
# Set a valid token.
# NEVER SAVE THE TOKEN HERE since this can trigger GIT err blocking commit and push!!!
export MS_TOKEN=YOUR_TOKEN_HERE

# Local mode (single machine)
python scripts/hub_upload/upload2hub.py \
--local \
--config ms.yaml

# Server mode (distribute tasks to clients) in tmux (recommended)
tmux new-session -d -s ms-upload-server "\
cd /home/rogerspyke/projects/robocoin-dataset && \
python scripts/hub_upload/upload2hub.py \
--server \
--config ms.yaml \
--host 0.0.0.0 \
--port 2101"
tmux attach -t ms-upload-server

# Client mode (connect to server and process tasks) in tmux (recommended)
tmux new-session -d -s ms-upload-client "\
cd /home/rogerspyke/projects/robocoin-dataset && \
python scripts/hub_upload/upload2hub.py \
--client \
--config ms.yaml \
--host 127.0.0.1 \
--port 2101 \
--num-clients 4"
tmux attach -t ms-upload-client
```

To avoid crashing between two hubs, use **different** server-client pairs and **different** ports for each hub (e.g. 2100 for HuggingFace, 2101 for ModelScope).