import carla

def main():
    client = None
    try:
        # Connect to the CARLA client
        client = carla.Client('localhost', 2000) # Default CARLA host and port
        client.set_timeout(10.0) # Set a timeout for the connection

        print(f"Connected to CARLA server version: {client.get_world().get_map().name}")

        # Get the blueprint library
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        print("\nAvailable vehicle blueprints:")
        for blueprint in blueprint_library.filter('vehicle.*'):
            print(f"- {blueprint.id}")

    except Exception as e:
        print(f"Error: Could not connect to CARLA server or retrieve blueprints. Is the CARLA server running? {e}")
    finally:
        if client:
            print("\nDisconnected from CARLA server.")

if __name__ == '__main__':
    main()
