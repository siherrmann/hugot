.PHONY: start-dev-container stop-dev-container run-tests clean help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

start-dev-container: ## Start the development container
	@echo "Starting development container..."
	./scripts/start-dev-container.sh

stop-dev-container: ## Stop the development container
	@echo "Stopping development container..."
	./scripts/stop-dev-container.sh

run-tests: ## Run all unit tests in container
	@echo "Running unit tests..."
	./scripts/run-unit-tests.sh

clean: ## Clean up test artifacts
	@echo "Cleaning up..."
	rm -rf testTarget artifacts

run-tests-container: ## Run unit tests inside a running container
	@echo "Running tests in container..."
	./scripts/run-unit-tests-container.sh

download-models: ## Download test models
	@echo "Downloading test models..."
	go run ./testData/downloadModels.go

install-cli: ## Install hugot CLI
	@echo "Installing hugot CLI..."
	./scripts/install-hugot-cli.sh
