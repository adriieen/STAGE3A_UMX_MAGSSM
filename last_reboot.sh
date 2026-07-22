#!/usr/bin/env bash
# Script pour afficher les 5 derniers reboots du système

echo "=== 5 Derniers Reboots du Système ==="
last reboot | head -n 5
