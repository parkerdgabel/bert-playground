"""Component scanner for automatic discovery and registration.

This module provides functionality to scan packages and modules for
decorated components and automatically register them with the container.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import List, Set, Type, Optional, Callable, Dict
from collections import defaultdict
import ast

from loguru import logger

from .decorators import (
    ComponentMetadata, ComponentType, get_component_metadata,
    get_registered_components
)
from .container import Container


class DependencyGraph:
    """Simple dependency graph for detecting circular dependencies."""
    
    def __init__(self):
        self.graph: Dict[Type, Set[Type]] = defaultdict(set)
        self.visited: Set[Type] = set()
        self.rec_stack: Set[Type] = set()
        
    def add_edge(self, from_node: Type, to_node: Type):
        """Add a dependency edge."""
        self.graph[from_node].add(to_node)
        
    def has_cycle(self) -> bool:
        """Check if the graph has a cycle."""
        for node in self.graph:
            if node not in self.visited:
                if self._has_cycle_util(node):
                    return True
        return False
        
    def _has_cycle_util(self, node: Type) -> bool:
        """Utility method for cycle detection using DFS."""
        self.visited.add(node)
        self.rec_stack.add(node)
        
        for neighbor in self.graph[node]:
            if neighbor not in self.visited:
                if self._has_cycle_util(neighbor):
                    return True
            elif neighbor in self.rec_stack:
                return True
                
        self.rec_stack.remove(node)
        return False
        
    def topological_sort(self) -> List[Type]:
        """Get topological ordering of nodes."""
        visited = set()
        stack = []
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in self.graph[node]:
                visit(neighbor)
            stack.append(node)
            
        for node in self.graph:
            visit(node)
            
        return stack[::-1]


class ComponentScanner:
    """Scanner for discovering and registering decorated components."""
    
    def __init__(self, container: Container):
        """Initialize the scanner.
        
        Args:
            container: The DI container to register components with
        """
        self.container = container
        self.discovered_components: List[Type] = []
        self.dependency_graph = DependencyGraph()
        self.active_profiles: Set[str] = set()
        
    def set_active_profiles(self, profiles: Set[str]):
        """Set active profiles for conditional registration.
        
        Args:
            profiles: Set of active profile names
        """
        self.active_profiles = profiles
        
    def scan_package(self, package_path: str) -> List[Type]:
        """Scan a package for decorated components.
        
        Args:
            package_path: Dotted path to the package (e.g., "domain.services")
            
        Returns:
            List of discovered component classes
        """
        logger.info(f"Scanning package: {package_path}")
        
        try:
            package = importlib.import_module(package_path)
        except ImportError as e:
            logger.warning(f"Failed to import package {package_path}: {e}")
            return []
            
        discovered = []
        
        # Walk through all modules in the package
        for _, module_name, is_pkg in pkgutil.walk_packages(
            package.__path__,
            package.__name__ + "."
        ):
            if not is_pkg:
                try:
                    module = importlib.import_module(module_name)
                    components = self._scan_module(module)
                    discovered.extend(components)
                except Exception as e:
                    logger.warning(f"Failed to scan module {module_name}: {e}")
                    
        self.discovered_components.extend(discovered)
        return discovered
        
    def scan_packages(self, package_paths: List[str]) -> List[Type]:
        """Scan multiple packages for components.
        
        Args:
            package_paths: List of package paths to scan
            
        Returns:
            List of all discovered component classes
        """
        all_components = []
        for package_path in package_paths:
            components = self.scan_package(package_path)
            all_components.extend(components)
        return all_components
        
    def scan_directory(self, directory: Path, base_package: str = "") -> List[Type]:
        """Scan a directory for Python files with decorated components.
        
        Args:
            directory: Directory path to scan
            base_package: Base package name for imports
            
        Returns:
            List of discovered component classes
        """
        discovered = []
        
        for py_file in directory.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue
                
            # Convert file path to module path
            relative_path = py_file.relative_to(directory)
            module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
            
            if base_package:
                module_name = f"{base_package}.{'.'.join(module_parts)}"
            else:
                module_name = '.'.join(module_parts)
                
            try:
                module = importlib.import_module(module_name)
                components = self._scan_module(module)
                discovered.extend(components)
            except Exception as e:
                logger.warning(f"Failed to scan file {py_file}: {e}")
                
        self.discovered_components.extend(discovered)
        return discovered
        
    def _scan_module(self, module) -> List[Type]:
        """Scan a module for decorated components.
        
        Args:
            module: The module to scan
            
        Returns:
            List of component classes found in the module
        """
        components = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip imported classes
            if obj.__module__ != module.__name__:
                continue
                
            # Check if it has DI metadata
            metadata = get_component_metadata(obj)
            if metadata:
                # Check if it should be registered
                if metadata.should_register(self.active_profiles):
                    components.append(obj)
                    logger.debug(
                        f"Discovered {metadata.component_type.value}: "
                        f"{obj.__name__} in {module.__name__}"
                    )
                else:
                    logger.debug(
                        f"Skipped {obj.__name__} due to profile/condition"
                    )
                    
        return components
        
    def register_all(self, validate: bool = True) -> None:
        """Register all discovered components with the container.
        
        Args:
            validate: Whether to validate dependencies before registration
        """
        if not self.discovered_components:
            logger.info("No components to register")
            return
            
        logger.info(f"Registering {len(self.discovered_components)} components")
        
        # Build dependency graph
        if validate:
            self._build_dependency_graph()
            
            # Check for circular dependencies
            if self.dependency_graph.has_cycle():
                raise RuntimeError("Circular dependency detected in components")
                
            # Get registration order
            registration_order = self.dependency_graph.topological_sort()
            
            # Register in dependency order
            registered = set()
            for component in registration_order:
                if component in self.discovered_components:
                    self._register_component(component)
                    registered.add(component)
                    
            # Register any remaining components not in the graph
            for component in self.discovered_components:
                if component not in registered:
                    self._register_component(component)
        else:
            # Register without validation
            for component in self.discovered_components:
                self._register_component(component)
                
    def _build_dependency_graph(self):
        """Build a dependency graph from discovered components."""
        for component in self.discovered_components:
            metadata = get_component_metadata(component)
            if metadata:
                for dependency in metadata.dependencies:
                    self.dependency_graph.add_edge(component, dependency)
                    
    def _register_component(self, component: Type):
        """Register a single component with the container.
        
        Args:
            component: The component class to register
        """
        try:
            self.container.register_decorator(component)
            logger.debug(f"Registered component: {component.__name__}")
        except Exception as e:
            logger.error(f"Failed to register {component.__name__}: {e}")
            raise
            
    def validate_architecture(self) -> List[str]:
        """Validate architectural constraints.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        for component in self.discovered_components:
            metadata = get_component_metadata(component)
            if not metadata:
                continue
                
            # Check dependency directions based on hexagonal architecture
            component_layer = self._get_layer(metadata.component_type)
            
            for dependency in metadata.dependencies:
                dep_metadata = get_component_metadata(dependency)
                if dep_metadata:
                    dep_layer = self._get_layer(dep_metadata.component_type)
                    
                    # Validate dependency direction
                    if not self._is_valid_dependency(component_layer, dep_layer):
                        errors.append(
                            f"{component.__name__} ({component_layer}) depends on "
                            f"{dependency.__name__} ({dep_layer}) - violates "
                            f"hexagonal architecture"
                        )
                        
        return errors
        
    def _get_layer(self, component_type: ComponentType) -> str:
        """Get the architectural layer for a component type.
        
        Args:
            component_type: The component type
            
        Returns:
            Layer name
        """
        if component_type in [ComponentType.DOMAIN_SERVICE, ComponentType.PORT]:
            return "domain"
        elif component_type in [ComponentType.APPLICATION_SERVICE, ComponentType.USE_CASE]:
            return "application"
        elif component_type in [ComponentType.ADAPTER, ComponentType.REPOSITORY, 
                                ComponentType.GATEWAY, ComponentType.CONTROLLER]:
            return "infrastructure"
        else:
            return "unknown"
            
    def _is_valid_dependency(self, from_layer: str, to_layer: str) -> bool:
        """Check if a dependency is valid according to hexagonal architecture.
        
        Args:
            from_layer: Source layer
            to_layer: Target layer
            
        Returns:
            True if dependency is valid
        """
        # Define allowed dependencies
        allowed = {
            "infrastructure": ["application", "domain"],
            "application": ["domain"],
            "domain": [],  # Domain should not depend on anything
            "unknown": ["domain", "application", "infrastructure"]
        }
        
        return to_layer in allowed.get(from_layer, [])
        
    def generate_report(self) -> str:
        """Generate a report of discovered components.
        
        Returns:
            Report string
        """
        report = ["Component Discovery Report", "=" * 50, ""]
        
        # Group by type
        by_type: Dict[ComponentType, List[Type]] = defaultdict(list)
        for component in self.discovered_components:
            metadata = get_component_metadata(component)
            if metadata:
                by_type[metadata.component_type].append(component)
                
        # Report by type
        for comp_type, components in by_type.items():
            report.append(f"{comp_type.value.upper()} ({len(components)})")
            report.append("-" * 30)
            for component in sorted(components, key=lambda c: c.__name__):
                metadata = get_component_metadata(component)
                scope = metadata.scope.value if metadata else "unknown"
                report.append(f"  - {component.__name__} (scope: {scope})")
            report.append("")
            
        # Summary
        report.append("SUMMARY")
        report.append("-" * 30)
        report.append(f"Total components: {len(self.discovered_components)}")
        
        return "\n".join(report)


def auto_discover_and_register(
    container: Container,
    package_paths: List[str],
    profiles: Optional[Set[str]] = None,
    validate: bool = True
) -> ComponentScanner:
    """Convenience function to scan and register components.
    
    Args:
        container: The DI container
        package_paths: List of package paths to scan
        profiles: Active profiles
        validate: Whether to validate dependencies
        
    Returns:
        The component scanner used
    """
    scanner = ComponentScanner(container)
    
    if profiles:
        scanner.set_active_profiles(profiles)
        
    scanner.scan_packages(package_paths)
    scanner.register_all(validate=validate)
    
    return scanner