#!/usr/bin/env python3
"""Validate container and runtime configuration compliance."""

import json
import subprocess
import sys
from pathlib import Path


def validate_docker_configuration():
    """Validate Docker and container configuration."""
    print("🐳 Validating Container & Runtime Configuration")
    print("=" * 60)

    compliance_results = {}

    # 1. Dockerfile Validation
    print("\n1. 📋 Dockerfile Configuration")
    dockerfile = Path("Dockerfile")

    if dockerfile.exists():
        content = dockerfile.read_text()

        # Check pinned base image
        has_pinned_image = "python:3.11.10-slim-bookworm" in content

        # Check non-root user
        has_nonroot_user = "USER trader" in content and "groupadd --gid 1000 trader" in content

        # Check health check
        has_healthcheck = "HEALTHCHECK" in content and "/health" in content

        # Check multi-stage build
        has_multistage = "FROM python:3.11.10-slim-bookworm AS builder" in content

        # Check exposed ports
        has_port_docs = "EXPOSE 5000 8001 8000" in content

        compliance_results["dockerfile_exists"] = True
        compliance_results["pinned_base_image"] = has_pinned_image
        compliance_results["nonroot_user"] = has_nonroot_user
        compliance_results["healthcheck"] = has_healthcheck
        compliance_results["multistage_build"] = has_multistage
        compliance_results["documented_ports"] = has_port_docs

        print(f"   ✅ Dockerfile exists")
        print(
            f"   {'✅' if has_pinned_image else '❌'} Pinned base image (python:3.11.10-slim-bookworm)"
        )
        print(f"   {'✅' if has_nonroot_user else '❌'} Non-root user configured (trader:1000)")
        print(f"   {'✅' if has_healthcheck else '❌'} Health check implemented")
        print(f"   {'✅' if has_multistage else '❌'} Multi-stage build for security")
        print(f"   {'✅' if has_port_docs else '❌'} Ports documented (5000, 8001, 8000)")
    else:
        compliance_results["dockerfile_exists"] = False
        print("   ❌ Dockerfile missing")

    # 2. Docker Compose Configuration
    print("\n2. 🔧 Docker Compose Setup")
    compose_file = Path("docker-compose.yml")

    if compose_file.exists():
        content = compose_file.read_text()

        # Check service configuration
        has_main_service = "cryptosmarttrader:" in content
        has_monitoring = "prometheus:" in content and "grafana:" in content

        # Check health checks
        has_compose_healthcheck = "healthcheck:" in content and "/health" in content

        # Check resource limits
        has_resource_limits = "deploy:" in content and "resources:" in content

        # Check volumes
        has_persistent_volumes = "./data:/app/data" in content and "./logs:/app/logs" in content

        # Check environment
        has_env_file = "env_file:" in content and ".env" in content

        compliance_results["compose_exists"] = True
        compliance_results["main_service"] = has_main_service
        compliance_results["monitoring_stack"] = has_monitoring
        compliance_results["compose_healthcheck"] = has_compose_healthcheck
        compliance_results["resource_limits"] = has_resource_limits
        compliance_results["persistent_volumes"] = has_persistent_volumes
        compliance_results["env_configuration"] = has_env_file

        print(f"   ✅ Docker Compose file exists")
        print(f"   {'✅' if has_main_service else '❌'} Main service configured")
        print(f"   {'✅' if has_monitoring else '❌'} Monitoring stack (Prometheus + Grafana)")
        print(f"   {'✅' if has_compose_healthcheck else '❌'} Health checks configured")
        print(f"   {'✅' if has_resource_limits else '❌'} Resource limits defined")
        print(f"   {'✅' if has_persistent_volumes else '❌'} Persistent volumes mounted")
        print(f"   {'✅' if has_env_file else '❌'} Environment file configuration")
    else:
        compliance_results["compose_exists"] = False
        print("   ❌ Docker Compose file missing")

    # 3. Kubernetes Configuration
    print("\n3. ☸️ Kubernetes Manifests")
    k8s_dir = Path("k8s")

    if k8s_dir.exists():
        # Check required manifests
        required_manifests = [
            "namespace.yaml",
            "configmap.yaml",
            "secret.yaml",
            "deployment.yaml",
            "service.yaml",
            "pvc.yaml",
        ]

        manifests_present = 0
        for manifest in required_manifests:
            if (k8s_dir / manifest).exists():
                manifests_present += 1
                print(f"   ✅ {manifest}")
            else:
                print(f"   ❌ {manifest} missing")

        # Check deployment configuration
        deployment_file = k8s_dir / "deployment.yaml"
        if deployment_file.exists():
            content = deployment_file.read_text()

            has_probes = all(
                probe in content for probe in ["livenessProbe:", "readinessProbe:", "startupProbe:"]
            )
            has_security = "runAsNonRoot: true" in content and "runAsUser: 1000" in content
            has_resources = "resources:" in content and "limits:" in content

            print(f"   {'✅' if has_probes else '❌'} All health probes configured")
            print(f"   {'✅' if has_security else '❌'} Security context (non-root)")
            print(f"   {'✅' if has_resources else '❌'} Resource limits defined")

        compliance_results["k8s_manifests"] = manifests_present >= 5
        compliance_results["k8s_complete"] = manifests_present == 6
    else:
        compliance_results["k8s_manifests"] = False
        print("   ❌ Kubernetes directory missing")

    # 4. Environment Configuration
    print("\n4. ⚙️ Environment & Settings")

    # Check Pydantic settings files
    config_files = ["config.json", ".env.example"]

    config_compliance = {}
    for config_file in config_files:
        exists = Path(config_file).exists()
        config_compliance[config_file] = exists
        print(f"   {'✅' if exists else '❌'} {config_file}")

    # Check for fail-fast configuration patterns in codebase
    pydantic_files = list(Path(".").rglob("*settings*.py")) + list(Path(".").rglob("*config*.py"))
    has_pydantic_settings = False

    for file_path in pydantic_files:
        if file_path.exists():
            content = file_path.read_text()
            if "BaseSettings" in content or "pydantic" in content:
                has_pydantic_settings = True
                break

    print(f"   {'✅' if has_pydantic_settings else '❌'} Pydantic Settings implementation")

    compliance_results["env_examples"] = config_compliance.get(".env.example", False)
    compliance_results["pydantic_settings"] = has_pydantic_settings

    # 5. Deployment Automation
    print("\n5. 🚀 Deployment Scripts")

    deployment_scripts = ["scripts/deploy.sh", "README_DOCKER.md"]

    script_compliance = {}
    for script in deployment_scripts:
        script_path = Path(script)
        exists = script_path.exists()
        script_compliance[script] = exists

        if exists and script.endswith(".sh"):
            # Check if executable
            import stat

            is_executable = script_path.stat().st_mode & stat.S_IEXEC
            print(
                f"   {'✅' if is_executable else '❌'} {script} (executable: {bool(is_executable)})"
            )
        else:
            print(f"   {'✅' if exists else '❌'} {script}")

    compliance_results["deployment_scripts"] = all(script_compliance.values())

    # 6. Monitoring Configuration
    print("\n6. 📊 Monitoring & Observability")

    monitoring_files = ["config/prometheus.yml", "config/prometheus/alert_rules.yml"]

    monitoring_compliance = {}
    for mon_file in monitoring_files:
        exists = Path(mon_file).exists()
        monitoring_compliance[mon_file] = exists
        print(f"   {'✅' if exists else '❌'} {mon_file}")

    compliance_results["monitoring_config"] = all(monitoring_compliance.values())

    # Summary
    print("\n" + "=" * 60)
    print("📊 CONTAINER & RUNTIME COMPLIANCE SUMMARY")
    print("=" * 60)

    total_checks = len(compliance_results)
    passed_checks = sum(1 for passed in compliance_results.values() if passed)
    compliance_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0

    for check, passed in compliance_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check.replace('_', ' ').title()}")

    print(f"\n🎯 OVERALL COMPLIANCE: {passed_checks}/{total_checks} ({compliance_rate:.1f}%)")

    if compliance_rate >= 90:
        print("🏆 EXCELLENT - Enterprise container standards fully met")
        return 0
    elif compliance_rate >= 80:
        print("✅ GOOD - Most container requirements met")
        return 1
    else:
        print("⚠️ NEEDS IMPROVEMENT - Container configuration gaps detected")
        return 2


def test_docker_build():
    """Test Docker build process without actually building."""
    print("\n🔧 Testing Docker Build Configuration")
    print("=" * 40)

    try:
        # Test Dockerfile syntax by doing a dry-run parse
        result = subprocess.run(
            ["docker", "build", "--dry-run", "."], capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            print("✅ Dockerfile syntax valid")
            return True
        else:
            print(f"❌ Dockerfile syntax error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ Docker build test timed out")
        return False
    except FileNotFoundError:
        print("⚠️ Docker not available for build testing")
        return False
    except Exception as e:
        print(f"❌ Docker build test failed: {e}")
        return False


if __name__ == "__main__":
    # Run validation
    exit_code = validate_docker_configuration()

    # Test Docker build if available
    docker_test_passed = test_docker_build()

    # Adjust exit code based on Docker test
    if exit_code == 0 and not docker_test_passed:
        exit_code = 1

    exit(exit_code)
