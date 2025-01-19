from vad import VAD

def main():
    vad = VAD()
    try:
        vad.run()
    except KeyboardInterrupt:
        print("\n程序已停止")

if __name__ == "__main__":
    main() 