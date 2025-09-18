#!/usr/bin/env python3
"""
GitHub Actions Local Testing Script
Tests GitHub Actions workflows locally using act or validation
"""
import os
import subprocess
import yaml
import json
from pathlib import Path
from typing import Dict, List

class GitHubActionsValidator:
    def __init__(self):
        self.workflows_dir = Path(".github/workflows")
        self.validation_results = {}
        
    def validate_workflow_syntax(self, workflow_file: Path) -> Dict:
        """Validate YAML syntax and structure of workflow file"""
        print(f"🔍 Validating {workflow_file.name}...")
        
        try:
            with open(workflow_file, 'r') as f:
                workflow = yaml.safe_load(f)
            
            if not workflow:
                return {
                    "valid": False,
                    "error": "Empty or invalid YAML file"
                }
            
            # Check required fields (handle 'on' keyword issue)
            required_fields = ['name', 'jobs']
            has_on_trigger = 'on' in workflow or True in workflow  # 'on' becomes True in YAML
            
            missing_fields = [field for field in required_fields if field not in workflow]
            if not has_on_trigger:
                missing_fields.append('on')
            
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required fields: {missing_fields}"
                }
            
            # Validate jobs structure
            jobs = workflow.get('jobs', {})
            if not isinstance(jobs, dict) or not jobs:
                return {
                    "valid": False,
                    "error": "No jobs defined or invalid jobs structure"
                }
            
            # Check each job
            job_issues = []
            for job_name, job_config in jobs.items():
                if 'runs-on' not in job_config:
                    job_issues.append(f"Job '{job_name}' missing 'runs-on'")
                
                if 'steps' not in job_config:
                    job_issues.append(f"Job '{job_name}' missing 'steps'")
            
            if job_issues:
                return {
                    "valid": False,
                    "error": "; ".join(job_issues)
                }
            
            return {
                "valid": True,
                "jobs_count": len(jobs),
                "triggers": list(workflow.get('on', workflow.get(True, {})).keys()) if isinstance(workflow.get('on', workflow.get(True, {})), dict) else [workflow.get('on', workflow.get(True))]
            }
            
        except yaml.YAMLError as e:
            return {
                "valid": False,
                "error": f"YAML syntax error: {e}"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {e}"
            }
    
    def check_required_secrets(self, workflow_file: Path) -> Dict:
        """Check for required secrets in workflow"""
        with open(workflow_file, 'r') as f:
            content = f.read()
        
        # Common secret patterns
        secret_patterns = [
            'secrets.GITHUB_TOKEN',
            'secrets.AZURE_CREDENTIALS',
            'secrets.DOCKER_USERNAME',
            'secrets.DOCKER_PASSWORD',
            'secrets.POSTGRES_URL',
            'secrets.MONGO_URL',
            'secrets.SECRET_KEY',
            'secrets.JWT_SECRET_KEY',
            'secrets.AZURE_STORAGE_CONNECTION_STRING',
            'secrets.DVC_AZURE_CONNECTION_STRING'
        ]
        
        found_secrets = []
        for pattern in secret_patterns:
            if pattern in content:
                found_secrets.append(pattern.replace('secrets.', ''))
        
        return {
            "required_secrets": found_secrets,
            "secrets_count": len(found_secrets)
        }
    
    def check_environment_variables(self, workflow_file: Path) -> Dict:
        """Check environment variables usage"""
        with open(workflow_file, 'r') as f:
            workflow = yaml.safe_load(f)
        
        env_vars = set()
        
        # Check global env
        if 'env' in workflow:
            env_vars.update(workflow['env'].keys())
        
        # Check job-level env
        for job in workflow.get('jobs', {}).values():
            if 'env' in job:
                env_vars.update(job['env'].keys())
            
            # Check step-level env
            for step in job.get('steps', []):
                if 'env' in step:
                    env_vars.update(step['env'].keys())
        
        return {
            "environment_variables": list(env_vars),
            "env_count": len(env_vars)
        }
    
    def validate_docker_references(self, workflow_file: Path) -> Dict:
        """Check Docker-related configurations"""
        with open(workflow_file, 'r') as f:
            content = f.read()
        
        docker_actions = [
            'docker/setup-buildx-action',
            'docker/login-action',
            'docker/build-push-action'
        ]
        
        used_actions = []
        for action in docker_actions:
            if action in content:
                used_actions.append(action)
        
        return {
            "docker_actions": used_actions,
            "uses_docker": len(used_actions) > 0
        }
    
    def check_azure_integration(self, workflow_file: Path) -> Dict:
        """Check Azure-specific integrations"""
        with open(workflow_file, 'r') as f:
            content = f.read()
        
        azure_actions = [
            'azure/login',
            'azure/aci-deploy',
            'azure/setup-kubectl'
        ]
        
        azure_services = [
            'Azure Container Instances',
            'Azure Container Registry',
            'Azure Storage',
            'Azure PostgreSQL'
        ]
        
        used_actions = [action for action in azure_actions if action in content]
        mentioned_services = [service for service in azure_services if service.lower() in content.lower()]
        
        return {
            "azure_actions": used_actions,
            "azure_services": mentioned_services,
            "uses_azure": len(used_actions) > 0
        }
    
    def validate_all_workflows(self) -> Dict:
        """Validate all workflow files"""
        print("🔍 Validating GitHub Actions workflows...")
        
        if not self.workflows_dir.exists():
            return {
                "status": "error",
                "message": "No .github/workflows directory found"
            }
        
        workflow_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))
        
        if not workflow_files:
            return {
                "status": "error",
                "message": "No workflow files found"
            }
        
        results = {}
        overall_valid = True
        
        for workflow_file in workflow_files:
            filename = workflow_file.name
            print(f"\n📋 Analyzing {filename}...")
            
            # Syntax validation
            syntax_result = self.validate_workflow_syntax(workflow_file)
            
            # Additional checks
            secrets_result = self.check_required_secrets(workflow_file)
            env_result = self.check_environment_variables(workflow_file)
            docker_result = self.validate_docker_references(workflow_file)
            azure_result = self.check_azure_integration(workflow_file)
            
            workflow_result = {
                "syntax": syntax_result,
                "secrets": secrets_result,
                "environment": env_result,
                "docker": docker_result,
                "azure": azure_result,
                "file_path": str(workflow_file)
            }
            
            results[filename] = workflow_result
            
            if not syntax_result["valid"]:
                overall_valid = False
                print(f"❌ {filename}: {syntax_result['error']}")
            else:
                print(f"✅ {filename}: Valid syntax")
                print(f"   Jobs: {syntax_result['jobs_count']}")
                print(f"   Triggers: {', '.join(syntax_result['triggers'])}")
                print(f"   Secrets: {secrets_result['secrets_count']}")
                print(f"   Env vars: {env_result['env_count']}")
                if docker_result['uses_docker']:
                    print(f"   Docker: {', '.join(docker_result['docker_actions'])}")
                if azure_result['uses_azure']:
                    print(f"   Azure: {', '.join(azure_result['azure_actions'])}")
        
        return {
            "status": "success" if overall_valid else "error",
            "workflows": results,
            "total_workflows": len(workflow_files),
            "valid_workflows": sum(1 for r in results.values() if r["syntax"]["valid"])
        }
    
    def generate_secrets_documentation(self, validation_results: Dict):
        """Generate documentation for required secrets"""
        all_secrets = set()
        
        for workflow_name, workflow_data in validation_results.get("workflows", {}).items():
            secrets = workflow_data.get("secrets", {}).get("required_secrets", [])
            all_secrets.update(secrets)
        
        if not all_secrets:
            return "No secrets required."
        
        docs = "# Required GitHub Secrets\n\n"
        docs += "Configure these secrets in your GitHub repository settings:\n\n"
        
        secret_descriptions = {
            "GITHUB_TOKEN": "GitHub personal access token (usually auto-provided)",
            "AZURE_CREDENTIALS": "Azure service principal credentials for deployment",
            "DOCKER_USERNAME": "Docker registry username",
            "DOCKER_PASSWORD": "Docker registry password or token",
            "POSTGRES_URL": "PostgreSQL database connection string",
            "MONGO_URL": "MongoDB connection string",
            "SECRET_KEY": "Application secret key for encryption",
            "JWT_SECRET_KEY": "JWT token signing secret",
            "AZURE_STORAGE_CONNECTION_STRING": "Azure Storage account connection string",
            "DVC_AZURE_CONNECTION_STRING": "DVC Azure storage connection string"
        }
        
        for secret in sorted(all_secrets):
            description = secret_descriptions.get(secret, "Description needed")
            docs += f"- **{secret}**: {description}\n"
        
        return docs
    
    def check_act_availability(self) -> bool:
        """Check if act (GitHub Actions local runner) is available"""
        try:
            result = subprocess.run(['act', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def run_local_test_with_act(self, workflow_name: str = None):
        """Run workflow locally using act (if available)"""
        if not self.check_act_availability():
            print("⚠️ 'act' is not installed. Cannot run local workflow tests.")
            print("Install act: https://github.com/nektos/act")
            return False
        
        try:
            cmd = ['act', '--dry-run']
            if workflow_name:
                cmd.extend(['--workflow', workflow_name])
            
            print(f"🏃 Running local test with act...")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                print("✅ Local workflow test successful")
                return True
            else:
                print(f"❌ Local workflow test failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running act: {e}")
            return False

def main():
    """Main validation runner"""
    print("⚙️ GitHub Actions Workflow Validator")
    print("=" * 40)
    
    validator = GitHubActionsValidator()
    
    # Validate all workflows
    results = validator.validate_all_workflows()
    
    if results["status"] == "error" and "message" in results:
        print(f"❌ {results['message']}")
        return False
    
    # Summary
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Total workflows: {results['total_workflows']}")
    print(f"Valid workflows: {results['valid_workflows']}")
    
    if results['total_workflows'] > 0:
        success_rate = (results['valid_workflows'] / results['total_workflows']) * 100
        print(f"Success rate: {success_rate:.1f}%")
    else:
        success_rate = 0
        print("No workflows found")
    
    # Generate secrets documentation
    print(f"\n📋 REQUIRED SECRETS")
    print("=" * 20)
    secrets_docs = validator.generate_secrets_documentation(results)
    print(secrets_docs)
    
    # Check for act and offer local testing
    if validator.check_act_availability():
        print(f"\n🏃 LOCAL TESTING")
        print("=" * 15)
        print("✅ 'act' is available for local workflow testing")
        
        # Run dry-run test
        validator.run_local_test_with_act()
    else:
        print(f"\n⚠️ LOCAL TESTING")
        print("=" * 15)
        print("'act' not found. Install for local workflow testing:")
        print("  macOS: brew install act")
        print("  Linux: https://github.com/nektos/act#installation")
    
    # Final assessment
    print(f"\n🎯 ASSESSMENT")
    print("=" * 12)
    
    if success_rate == 100:
        print("🎉 All workflows are valid and ready for use!")
        return True
    elif success_rate >= 80:
        print("✅ Most workflows are valid. Fix remaining issues.")
        return True
    else:
        print("❌ Multiple workflow issues found. Review and fix.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
