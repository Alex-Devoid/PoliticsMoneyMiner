from watchfiles import watch
import subprocess

def run_tests():
    print("Running tests...")
    subprocess.run(["python", "../tests/debug_test.py"], check=True)

def main():
    print("Watching for file changes...")
    for changes in watch("../cloud_function"):
        print(f"Detected changes: {changes}")
        run_tests()

if __name__ == "__main__":
    main()
