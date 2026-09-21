FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir '.[classic]' \
    && useradd --create-home arcade && mkdir runs && chown arcade runs
USER arcade
EXPOSE 8000
CMD ["jev-arcade", "serve", "--host", "0.0.0.0"]
