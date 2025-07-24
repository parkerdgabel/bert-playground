"""Enhanced decorators for dependency injection with hexagonal architecture support.

This module provides modern decorator patterns for component registration,
automatic discovery, and architectural layer enforcement.
"""

from typing import (
    Type, Optional, Any, Dict, List, Set, Callable, Union, TypeVar, 
    get_type_hints, get_origin, get_args
)
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
import inspect
from threading import Lock

from loguru import logger

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class ComponentType(Enum):
    """Types of components in the hexagonal architecture."""
    DOMAIN_SERVICE = "domain_service"
    APPLICATION_SERVICE = "application_service"
    USE_CASE = "use_case"
    PORT = "port"
    ADAPTER = "adapter"
    REPOSITORY = "repository"
    FACTORY = "factory"
    HANDLER = "handler"
    CONTROLLER = "controller"
    GATEWAY = "gateway"


class Scope(Enum):
    """Component lifecycle scopes."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    REQUEST = "request"
    SESSION = "session"


@dataclass
class ComponentMetadata:
    """Metadata for registered components."""
    component_type: ComponentType
    name: Optional[str] = None
    scope: Scope = Scope.TRANSIENT
    priority: int = 0
    tags: Set[str] = field(default_factory=set)
    lazy: bool = False
    interfaces: List[Type] = field(default_factory=list)
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[Type] = field(default_factory=list)
    port_type: Optional[Type] = None  # For adapters
    profiles: Set[str] = field(default_factory=set)
    conditions: List[Callable[[], bool]] = field(default_factory=list)
    init_method: Optional[str] = None
    destroy_method: Optional[str] = None
    
    def should_register(self, active_profiles: Optional[Set[str]] = None) -> bool:
        """Check if component should be registered based on conditions."""
        # Check profiles
        if self.profiles:
            active = active_profiles or set()
            if not self.profiles.intersection(active):
                return False
                
        # Check conditions
        for condition in self.conditions:
            try:
                if not condition():
                    return False
            except Exception as e:
                logger.warning(f"Condition check failed: {e}")
                return False
                
        return True


# Global registry for decorated components
_component_registry: Dict[Type, ComponentMetadata] = {}
_registry_lock = Lock()


def get_component_metadata(cls: Type) -> Optional[ComponentMetadata]:
    """Get metadata for a component class."""
    return _component_registry.get(cls)


def _extract_interfaces(cls: Type) -> List[Type]:
    """Extract interfaces (protocols/ABCs) implemented by a class."""
    interfaces = []
    
    # Get direct bases that are protocols or ABCs
    for base in cls.__bases__:
        if base is object:
            continue
            
        # Check if it's a Protocol
        if hasattr(base, "_is_protocol") and base._is_protocol:
            interfaces.append(base)
        # Check if it's an ABC
        elif hasattr(base, "__abstractmethods__") and base.__abstractmethods__:
            interfaces.append(base)
        # Check if it looks like an interface (ends with Protocol, Port, etc.)
        elif any(base.__name__.endswith(suffix) for suffix in ["Protocol", "Port", "Interface", "Service"]):
            interfaces.append(base)
            
    return interfaces


def _extract_dependencies(cls: Type) -> List[Type]:
    """Extract constructor dependencies from a class."""
    dependencies = []
    
    try:
        # Get constructor signature
        init_method = cls.__init__
        sig = inspect.signature(init_method)
        
        # Extract parameter types
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            if param.annotation != param.empty:
                param_type = param.annotation
                
                # Handle Optional types
                origin = get_origin(param_type)
                if origin is Union:
                    args = get_args(param_type)
                    # Filter out None from Union types (Optional)
                    non_none_types = [t for t in args if t is not type(None)]
                    if non_none_types:
                        param_type = non_none_types[0]
                        
                dependencies.append(param_type)
                
    except (AttributeError, ValueError):
        pass
        
    return dependencies


def _register_component(cls: Type, metadata: ComponentMetadata) -> None:
    """Register a component with its metadata."""
    with _registry_lock:
        _component_registry[cls] = metadata
        
        # Also store metadata on the class itself
        cls._di_metadata = metadata
        
    logger.debug(
        f"Registered {metadata.component_type.value} component: {cls.__name__} "
        f"(scope={metadata.scope.value}, priority={metadata.priority})"
    )


# Base component decorator
def component(
    component_type: ComponentType = ComponentType.DOMAIN_SERVICE,
    *,
    name: Optional[str] = None,
    scope: Union[Scope, str] = Scope.TRANSIENT,
    priority: int = 0,
    tags: Optional[Union[List[str], Set[str]]] = None,
    lazy: bool = False,
    bind_to: Optional[Union[Type, List[Type]]] = None,
    profiles: Optional[Union[str, List[str]]] = None,
    condition: Optional[Callable[[], bool]] = None,
    init_method: Optional[str] = None,
    destroy_method: Optional[str] = None,
    **kwargs
) -> Callable[[Type[T]], Type[T]]:
    """
    Base decorator for marking components for dependency injection.
    
    Args:
        component_type: Type of component in the architecture
        name: Optional name for the component
        scope: Lifecycle scope (singleton, transient, etc.)
        priority: Priority for resolution when multiple implementations exist
        tags: Tags for grouping/filtering components
        lazy: Whether to initialize lazily
        bind_to: Interface(s) to bind this implementation to
        profiles: Profile(s) in which this component is active
        condition: Condition function that must return True for registration
        init_method: Method name to call after construction
        destroy_method: Method name to call before destruction
        **kwargs: Additional metadata
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Normalize scope
        if isinstance(scope, str):
            scope_enum = Scope(scope)
        else:
            scope_enum = scope
            
        # Normalize interfaces
        interfaces = []
        if bind_to:
            if isinstance(bind_to, list):
                interfaces.extend(bind_to)
            else:
                interfaces.append(bind_to)
        
        # Auto-detect interfaces if not specified
        if not interfaces:
            interfaces = _extract_interfaces(cls)
            
        # Extract dependencies
        dependencies = _extract_dependencies(cls)
        
        # Normalize tags
        tag_set = set(tags) if tags else set()
        
        # Normalize profiles
        profile_set = set()
        if profiles:
            if isinstance(profiles, str):
                profile_set.add(profiles)
            else:
                profile_set.update(profiles)
                
        # Build conditions list
        conditions = []
        if condition:
            conditions.append(condition)
            
        # Create metadata
        metadata = ComponentMetadata(
            component_type=component_type,
            name=name or cls.__name__,
            scope=scope_enum,
            priority=priority,
            tags=tag_set,
            lazy=lazy,
            interfaces=interfaces,
            qualifiers=kwargs,
            dependencies=dependencies,
            profiles=profile_set,
            conditions=conditions,
            init_method=init_method,
            destroy_method=destroy_method,
        )
        
        # Register component
        _register_component(cls, metadata)
        
        # Mark class as injectable (for backward compatibility)
        cls._injectable = True
        
        return cls
        
    return decorator


# Helper for dual-syntax decorators
def _make_decorator(
    component_type: ComponentType,
    default_scope: Union[Scope, str],
    tags: Set[str],
    cls: Optional[Type[T]] = None,
    *,
    name: Optional[str] = None,
    scope: Optional[Union[Scope, str]] = None,
    **kwargs
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """Helper to create decorators that support both @decorator and @decorator() syntax."""
    if scope is None:
        scope = default_scope
        
    def decorator(cls: Type[T]) -> Type[T]:
        return component(
            component_type,
            name=name,
            scope=scope,
            tags=tags,
            **kwargs
        )(cls)
    
    if cls is None:
        return decorator
    else:
        return decorator(cls)


# Layer-specific decorators

def service(
    cls: Optional[Type[T]] = None,
    *,
    name: Optional[str] = None,
    scope: Union[Scope, str] = Scope.SINGLETON,
    **kwargs
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """
    Decorator for domain services.
    
    Domain services contain pure business logic without framework dependencies.
    They are singleton by default.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        return component(
            ComponentType.DOMAIN_SERVICE,
            name=name,
            scope=scope,
            tags={"domain", "service"},
            **kwargs
        )(cls)
    
    if cls is None:
        # Called with parameters: @service(...)
        return decorator
    else:
        # Called without parameters: @service
        return decorator(cls)


def application_service(
    name: Optional[str] = None,
    *,
    scope: Union[Scope, str] = Scope.TRANSIENT,
    **kwargs
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for application services.
    
    Application services orchestrate domain services and handle use case flow.
    They are transient by default.
    """
    return component(
        ComponentType.APPLICATION_SERVICE,
        name=name,
        scope=scope,
        tags={"application", "service"},
        **kwargs
    )


def use_case(
    cls: Optional[Type[T]] = None,
    *,
    name: Optional[str] = None,
    scope: Union[Scope, str] = Scope.TRANSIENT,
    **kwargs
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """
    Decorator for use cases.
    
    Use cases represent specific application operations.
    They are transient by default.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        return component(
            ComponentType.USE_CASE,
            name=name,
            scope=scope,
            tags={"application", "use_case"},
            **kwargs
        )(cls)
    
    if cls is None:
        return decorator
    else:
        return decorator(cls)


def port(
    name: Optional[str] = None,
    **kwargs
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for ports (interfaces).
    
    Ports define contracts between layers in hexagonal architecture.
    """
    return component(
        ComponentType.PORT,
        name=name,
        tags={"port", "interface"},
        **kwargs
    )


def adapter(
    port_type: Type,
    *,
    name: Optional[str] = None,
    scope: Union[Scope, str] = Scope.SINGLETON,
    priority: int = 0,
    **kwargs
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for adapters.
    
    Adapters implement ports and connect to external systems.
    They are singleton by default.
    
    Args:
        port_type: The port interface this adapter implements
        name: Optional adapter name
        scope: Lifecycle scope
        priority: Priority for selection when multiple adapters exist
        **kwargs: Additional metadata
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Apply component decorator
        component(
            ComponentType.ADAPTER,
            name=name,
            scope=scope,
            priority=priority,
            bind_to=port_type,
            tags={"adapter", "infrastructure"},
            **kwargs
        )(cls)
        
        # Add port type to metadata
        metadata = get_component_metadata(cls)
        if metadata:
            metadata.port_type = port_type
            
        return cls
        
    return decorator


def repository(
    cls: Optional[Type[T]] = None,
    *,
    name: Optional[str] = None,
    scope: Union[Scope, str] = Scope.SINGLETON,
    **kwargs
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """
    Decorator for repositories.
    
    Repositories handle data persistence and retrieval.
    They are singleton by default.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        return component(
            ComponentType.REPOSITORY,
            name=name,
            scope=scope,
            tags={"repository", "infrastructure"},
            **kwargs
        )(cls)
    
    if cls is None:
        return decorator
    else:
        return decorator(cls)


def factory(
    produces: Type,
    *,
    name: Optional[str] = None,
    scope: Union[Scope, str] = Scope.SINGLETON,
    **kwargs
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for factory classes.
    
    Factories create instances of other objects.
    They are singleton by default.
    
    Args:
        produces: Type that this factory produces
        name: Optional factory name
        scope: Lifecycle scope
        **kwargs: Additional metadata
    """
    return component(
        ComponentType.FACTORY,
        name=name,
        scope=scope,
        bind_to=produces,
        tags={"factory"},
        **kwargs
    )


def handler(
    name: Optional[str] = None,
    *,
    scope: Union[Scope, str] = Scope.TRANSIENT,
    **kwargs
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for event/command handlers.
    
    Handlers process events or commands in the system.
    They are transient by default.
    """
    return component(
        ComponentType.HANDLER,
        name=name,
        scope=scope,
        tags={"handler"},
        **kwargs
    )


def controller(
    name: Optional[str] = None,
    *,
    scope: Union[Scope, str] = Scope.TRANSIENT,
    **kwargs
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for controllers.
    
    Controllers handle external requests (HTTP, CLI, etc.).
    They are transient by default.
    """
    return component(
        ComponentType.CONTROLLER,
        name=name,
        scope=scope,
        tags={"controller", "primary"},
        **kwargs
    )


def gateway(
    name: Optional[str] = None,
    *,
    scope: Union[Scope, str] = Scope.SINGLETON,
    **kwargs
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for gateways.
    
    Gateways handle communication with external systems.
    They are singleton by default.
    """
    return component(
        ComponentType.GATEWAY,
        name=name,
        scope=scope,
        tags={"gateway", "infrastructure"},
        **kwargs
    )


# Qualifier decorators

def qualifier(name: str) -> Any:
    """
    Create a qualifier for dependency injection.
    
    Used with Annotated to specify which implementation to inject:
    ```python
    def __init__(self, cache: Annotated[CachePort, qualifier("redis")]):
        pass
    ```
    """
    class Qualifier:
        def __init__(self):
            self.name = name
            
        def __repr__(self):
            return f"Qualifier({name!r})"
            
    return Qualifier()


def primary() -> Callable[[Type[T]], Type[T]]:
    """
    Mark a component as the primary implementation.
    
    When multiple implementations exist, the primary one is selected by default.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        metadata = get_component_metadata(cls)
        if metadata:
            metadata.priority = 1000  # High priority
            metadata.qualifiers["primary"] = True
        return cls
        
    return decorator


# Lifecycle decorators

def post_construct(method: F) -> F:
    """
    Mark a method to be called after construction.
    
    The method will be called after all dependencies are injected.
    """
    method._di_post_construct = True
    
    # Store method name in class metadata
    if hasattr(method, "__self__"):
        cls = method.__self__.__class__
        metadata = get_component_metadata(cls)
        if metadata:
            metadata.init_method = method.__name__
            
    return method


def pre_destroy(method: F) -> F:
    """
    Mark a method to be called before destruction.
    
    The method will be called when the component is being destroyed.
    """
    method._di_pre_destroy = True
    
    # Store method name in class metadata
    if hasattr(method, "__self__"):
        cls = method.__self__.__class__
        metadata = get_component_metadata(cls)
        if metadata:
            metadata.destroy_method = method.__name__
            
    return method


def lazy() -> Callable[[Type[T]], Type[T]]:
    """
    Mark a component for lazy initialization.
    
    The component will only be created when first accessed.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        metadata = get_component_metadata(cls)
        if metadata:
            metadata.lazy = True
        return cls
        
    return decorator


# Conditional decorators

def conditional(condition: Callable[[], bool]) -> Callable[[Type[T]], Type[T]]:
    """
    Register component only if condition is met.
    
    The condition function is evaluated at registration time.
    
    Args:
        condition: Function that returns True if component should be registered
    """
    def decorator(cls: Type[T]) -> Type[T]:
        metadata = get_component_metadata(cls)
        if metadata:
            metadata.conditions.append(condition)
        return cls
        
    return decorator


def profile(profile_name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Register component only for specific profile.
    
    Args:
        profile_name: Profile in which this component is active
    """
    def decorator(cls: Type[T]) -> Type[T]:
        metadata = get_component_metadata(cls)
        if metadata:
            metadata.profiles.add(profile_name)
        return cls
        
    return decorator


def profiles(*profile_names: str) -> Callable[[Type[T]], Type[T]]:
    """
    Register component for multiple profiles.
    
    Args:
        *profile_names: Profiles in which this component is active
    """
    def decorator(cls: Type[T]) -> Type[T]:
        metadata = get_component_metadata(cls)
        if metadata:
            metadata.profiles.update(profile_names)
        return cls
        
    return decorator


# Configuration decorators

def value(key: str, default: Any = None) -> Any:
    """
    Inject configuration value.
    
    Used with Annotated to inject configuration values:
    ```python
    def __init__(self, host: Annotated[str, value("db.host", "localhost")]):
        pass
    ```
    """
    class Value:
        def __init__(self):
            self.key = key
            self.default = default
            
        def __repr__(self):
            return f"Value({key!r}, default={default!r})"
            
    return Value()


# Utility functions

def get_registered_components(
    component_type: Optional[ComponentType] = None,
    tags: Optional[Set[str]] = None,
    profiles: Optional[Set[str]] = None
) -> List[Type]:
    """
    Get all registered components matching criteria.
    
    Args:
        component_type: Filter by component type
        tags: Filter by tags (must have all specified tags)
        profiles: Filter by profiles (must be active in at least one)
        
    Returns:
        List of component classes
    """
    components = []
    
    for cls, metadata in _component_registry.items():
        # Filter by type
        if component_type and metadata.component_type != component_type:
            continue
            
        # Filter by tags
        if tags and not tags.issubset(metadata.tags):
            continue
            
        # Filter by profiles
        if profiles and not metadata.profiles.intersection(profiles):
            continue
            
        components.append(cls)
        
    return components


def clear_registry() -> None:
    """Clear the component registry (mainly for testing)."""
    with _registry_lock:
        _component_registry.clear()