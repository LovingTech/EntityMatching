FROM quay.io/jupyter/datascience-notebook:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN mkdir /tmp
COPY ./uv.lock ./pyproject.toml /home/jovyan/
WORKDIR /home/jovyan
RUN uv sync --locked