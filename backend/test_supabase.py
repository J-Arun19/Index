import os
import sys
import uuid

# Add the directory containing this file to python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)
# Also add project root
project_root = os.path.dirname(backend_dir)
sys.path.append(project_root)

from backend.services.supabase_service import SupabaseService
from backend.services.database_service import DatabaseService

def create_mock_file(filename: str, content: str) -> str:
    path = os.path.join(backend_dir, filename)
    with open(path, "w") as f:
        f.write(content)
    return path

def main():
    print("==================================================")
    print(" SatyaDrishti Supabase Storage Test Suite ")
    print("==================================================")
    
    try:
        SupabaseService.initialize()
        print("[OK] Supabase client configuration initialized.")
    except Exception as e:
        print(f"[ERROR] Configuration Initialization Failed: {str(e)}")
        sys.exit(1)
        
    print("\nStarting connectivity handshake...")
    is_connected = SupabaseService.verify_connection()
    if not is_connected:
        print("[ERROR] Connection verification failed. Check environment variables.")
        sys.exit(1)
    print("[OK] Connection handshake succeeded.")

    # 1. Initialize Storage Buckets
    print("\nInitializing storage buckets...")
    DatabaseService.initialize_storage()
    print("[OK] Storage buckets verified/configured.")

    # Generate testing file IDs
    test_uuid = uuid.uuid4().hex[:8]
    
    mock_image_path = create_mock_file("mock_image.jpg", f"fake-jpeg-data-{test_uuid}")
    mock_video_path = create_mock_file("mock_video.mp4", f"fake-mpeg4-data-{test_uuid}")
    mock_pdf_path = create_mock_file("mock_report.pdf", f"fake-pdf-report-data-{test_uuid}")

    dl_image_path = os.path.join(backend_dir, "dl_mock_image.jpg")
    dl_video_path = os.path.join(backend_dir, "dl_mock_video.mp4")
    dl_pdf_path = os.path.join(backend_dir, "dl_mock_report.pdf")

    try:
        # A. Media Storage Tests (Upload / Download / Delete)
        print("\n[TEST A] TESTING MEDIA STORAGE OPERATIONS...")
        
        # 1. Upload Image
        image_dest = f"images/test_image_{test_uuid}.jpg"
        print(f"  Uploading image -> '{image_dest}'...")
        image_storage_path = SupabaseService.upload_media(mock_image_path, image_dest)
        if not image_storage_path:
            raise Exception("upload_media for image returned None.")
        print(f"  [OK] Image uploaded successfully. Path: '{image_storage_path}'")
        
        # 2. Upload Video
        video_dest = f"videos/test_video_{test_uuid}.mp4"
        print(f"  Uploading video -> '{video_dest}'...")
        video_storage_path = SupabaseService.upload_media(mock_video_path, video_dest)
        if not video_storage_path:
            raise Exception("upload_media for video returned None.")
        print(f"  [OK] Video uploaded successfully. Path: '{video_storage_path}'")

        # 3. Download Media
        print(f"  Downloading image from '{image_storage_path}'...")
        dl_img_ok = SupabaseService.download_media(image_storage_path, dl_image_path)
        if not dl_img_ok:
            raise Exception("download_media for image failed.")
        print("  [OK] Image downloaded successfully.")

        print(f"  Downloading video from '{video_storage_path}'...")
        dl_vid_ok = SupabaseService.download_media(video_storage_path, dl_video_path)
        if not dl_vid_ok:
            raise Exception("download_media for video failed.")
        print("  [OK] Video downloaded successfully.")

        # 4. Delete Media
        print(f"  Deleting image '{image_storage_path}'...")
        del_img_ok = SupabaseService.delete_media(image_storage_path)
        if not del_img_ok:
            raise Exception("delete_media for image failed.")
        print("  [OK] Image deleted from storage.")

        print(f"  Deleting video '{video_storage_path}'...")
        del_vid_ok = SupabaseService.delete_media(video_storage_path)
        if not del_vid_ok:
            raise Exception("delete_media for video failed.")
        print("  [OK] Video deleted from storage.")

        # B. Report Storage Tests (Upload / Download / Delete)
        print("\n[TEST B] TESTING REPORT STORAGE OPERATIONS...")
        
        # 1. Upload Report
        report_dest = f"report_test_{test_uuid}.pdf"
        print(f"  Uploading report -> '{report_dest}'...")
        report_storage_path = SupabaseService.upload_report(mock_pdf_path, report_dest)
        if not report_storage_path:
            raise Exception("upload_report returned None.")
        print(f"  [OK] Report uploaded successfully. Path: '{report_storage_path}'")

        # 2. Download Report
        print(f"  Downloading report from '{report_storage_path}'...")
        dl_pdf_ok = SupabaseService.download_report(report_storage_path, dl_pdf_path)
        if not dl_pdf_ok:
            raise Exception("download_report failed.")
        print("  [OK] Report downloaded successfully.")

        # 3. Delete Report
        print(f"  Deleting report '{report_storage_path}'...")
        del_pdf_ok = SupabaseService.delete_report(report_storage_path)
        if not del_pdf_ok:
            raise Exception("delete_report failed.")
        print("  [OK] Report deleted from storage.")

        print("\n==================================================")
        print(" [SUCCESS] ALL STORAGE FUNCTIONALITIES VERIFIED! ")
        print("==================================================")

    except Exception as e:
        print(f"\n[ERROR] Storage Verification Failed: {str(e)}")
        sys.exit(1)
        
    finally:
        # Clean up local mock files
        for p in [mock_image_path, mock_video_path, mock_pdf_path, dl_image_path, dl_video_path, dl_pdf_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

if __name__ == "__main__":
    main()
