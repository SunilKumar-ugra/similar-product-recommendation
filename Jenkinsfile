pipeline {
    agent any

    environment {
        DOCKER_BUILDKIT = '1'
        COMPOSE_PROJECT_NAME = 'similar-recommender'
        PYTHONUNBUFFERED = '1'
    }

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'Checking out source code...'
                checkout scm
            }
        }

        stage('Install Python Dependencies') {
            steps {
                echo 'Installing Python dependencies for artifact build...'
                sh '''
                    python3 -m venv .venv || true
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Build Artifacts (Embeddings + Index)') {
            steps {
                echo 'Building embeddings and FAISS index...'
                sh '''
                    . .venv/bin/activate
                    python entrypoints/build_artifacts.py
                '''
            }
        }

        stage('Build Docker Images') {
            steps {
                echo 'Building Docker images...'
                sh '''
                    docker compose build
                '''
            }
        }

        stage('Sanity Check (API)') {
            steps {
                echo 'Running API sanity check...'
                sh '''
                    docker compose up -d api
                    sleep 15
                    curl -f http://localhost:8000/docs || exit 1
                '''
            }
        }

        stage('Deploy (API + UI)') {
            steps {
                echo 'Deploying full application...'
                sh '''
                    docker compose down
                    docker compose up -d
                '''
            }
        }
    }

    post {
        success {
            echo '✅ Deployment completed successfully'
        }

        failure {
            echo '❌ Deployment failed'
        }

        always {
            echo 'Cleaning up dangling Docker resources'
            sh 'docker system prune -f || true'
        }
    }
}
