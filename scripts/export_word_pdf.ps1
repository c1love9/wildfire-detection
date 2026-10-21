param([Parameter(Mandatory=$true)][string]$InputPath,[Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference='Stop'
$documentApp=$null
$reportDocument=$null
try {
    # Create a dedicated invisible instance. Never attach to or close user documents.
    $documentApp=New-Object -ComObject Word.Application
    if ($documentApp.Documents.Count -ne 0) { throw 'The automation instance contains other documents; refusing to proceed.' }
    $documentApp.Visible=$false
    $documentApp.DisplayAlerts=0
    $documentApp.AutomationSecurity=3
    $reportDocument=$documentApp.Documents.Open([IO.Path]::GetFullPath($InputPath),$false,$true)
    $reportDocument.ExportAsFixedFormat([IO.Path]::GetFullPath($OutputPath),17)
    Write-Output 'Native Microsoft Word PDF export completed.'
}
finally {
    if ($null -ne $reportDocument) { $reportDocument.Close(0); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($reportDocument) }
    if ($null -ne $documentApp) {
        if ($documentApp.Documents.Count -eq 0) { $documentApp.Quit() }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($documentApp)
    }
}
