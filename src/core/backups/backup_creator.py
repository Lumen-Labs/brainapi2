"""
File: /backup_creator.py
Project: backups
Created Date: Monday January 26th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 26th 2026 9:43:51 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""


def create_backup(brain_id: str) -> Backup:
    """
    Create a backup of the knowledge graph identified by brain_id.
    
    Parameters:
        brain_id (str): Identifier of the brain (knowledge graph) to back up.
    
    Returns:
        Backup: Object containing metadata for the created backup and a download link or path to the stored backup.
    """

    # Create Graph backup
    # Create Vector Store backup
    # Create TextData/StructuredData backup
    # Create Observations backup

    # Store into s3 folder the files

    # Do inside a celery task

    # Use cache to track progress

    # Return the backup path download link object

    pass